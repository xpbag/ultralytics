import torch
import itertools

from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite
from timm.models.registry import register_model
from .ska import SKA

from timm.models.helpers import build_model_with_cfg
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = self.d * num_heads
        self.attn_ratio = attn_ratio

        # 1. 移除resolution参数，动态适配尺寸
        h = self.dh + nh_kd * 2
        self.qkv = Conv2d_BN(dim, h, ks=1)
        self.proj = nn.Sequential(nn.ReLU(), Conv2d_BN(self.dh, dim, bn_weight_init=1))  # 修复BN初始化
        self.dw = Conv2d_BN(nh_kd, nh_kd, 3, 1, 1, groups=nh_kd)

        # 2. 动态扩展的位置偏置参数（初始为空）
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, 0))
        # 推理模式缓存
        self._bias_cache = {}

    def _get_attention_bias(self, H, W):
        """动态生成当前H/W对应的位置偏置"""
        points = list(itertools.product(range(H), range(W)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        # 扩展偏置参数（适配新的偏移类型）
        num_new = len(attention_offsets)
        if self.attention_biases.shape[1] < num_new:
            new_biases = torch.zeros(self.num_heads, num_new - self.attention_biases.shape[1],
                                     device=self.attention_biases.device)
            nn.init.trunc_normal_(new_biases, std=0.02)  # 修复初始化
            self.attention_biases = nn.Parameter(torch.cat([self.attention_biases, new_biases], dim=1))
        # 生成偏置矩阵
        bias_idxs = torch.LongTensor(idxs).view(N, N).to(self.attention_biases.device)
        return self.attention_biases[:, bias_idxs]

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode:
            self._bias_cache.clear()  # 训练模式清空缓存
        else:
            pass  # 推理模式缓存在forward中生成

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # 1. 提取qkv（删除冗余view）
        qkv = self.qkv(x)
        q, k, v = qkv.split([self.nh_kd, self.nh_kd, self.dh], dim=1)

        # 2. q深度卷积增强
        q = self.dw(q)

        # 3. 维度重塑
        q = q.view(B, self.num_heads, self.key_dim, N)
        k = k.view(B, self.num_heads, self.key_dim, N)
        v = v.view(B, self.num_heads, self.d, N)

        # 4. 动态获取位置偏置（推理缓存）
        if not self.training:
            cache_key = (H, W)
            if cache_key not in self._bias_cache:
                self._bias_cache[cache_key] = self._get_attention_bias(H, W)
            attn_bias = self._bias_cache[cache_key]
        else:
            attn_bias = self._get_attention_bias(H, W)

        # 5. 注意力计算（显式广播batch维度）
        attn = (q.transpose(-2, -1) @ k) * self.scale + attn_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)

        # 6. 加权求和 + 投影
        x = (v @ attn.transpose(-2, -1)).reshape(B, -1, H, W)
        x = self.proj(x)

        return x
# class Attention(torch.nn.Module):
#     def __init__(self, dim, key_dim, num_heads=8,
#                  attn_ratio=4,
#                  resolution=14):
#         super().__init__()
#         self.num_heads = num_heads
#         self.scale = key_dim ** -0.5
#         self.key_dim = key_dim
#         self.nh_kd = nh_kd = key_dim * num_heads
#         self.d = int(attn_ratio * key_dim)
#         self.dh = int(attn_ratio * key_dim) * num_heads
#         self.attn_ratio = attn_ratio
#         h = self.dh + nh_kd * 2
#         self.qkv = Conv2d_BN(dim, h, ks=1)
#         self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
#             self.dh, dim, bn_weight_init=0))
#         self.dw = Conv2d_BN(nh_kd, nh_kd, 3, 1, 1, groups=nh_kd)
#         points = list(itertools.product(range(resolution), range(resolution)))
#         N = len(points)
#         attention_offsets = {}
#         idxs = []
#         for p1 in points:
#             for p2 in points:
#                 offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
#                 if offset not in attention_offsets:
#                     attention_offsets[offset] = len(attention_offsets)
#                 idxs.append(attention_offsets[offset])
#         self.attention_biases = torch.nn.Parameter(
#             torch.zeros(num_heads, len(attention_offsets)))
#         self.register_buffer('attention_bias_idxs',
#                              torch.LongTensor(idxs).view(N, N))
#
#     @torch.no_grad()
#     def train(self, mode=True):
#         super().train(mode)
#         if mode and hasattr(self, 'ab'):
#             del self.ab
#         else:
#             self.ab = self.attention_biases[:, self.attention_bias_idxs]
#
#     def forward(self, x):
#         B, _, H, W = x.shape
#         N = H * W
#         qkv = self.qkv(x)
#         q, k, v = qkv.view(B, -1, H, W).split([self.nh_kd, self.nh_kd, self.dh], dim=1)
#         q = self.dw(q)
#         q, k, v = q.view(B, self.num_heads, -1, N), k.view(B, self.num_heads, -1, N), v.view(B, self.num_heads, -1, N)
#
#         print("q.shape = ", q.shape)
#         print("k.shape = ", k.shape)
#         print("v.shape = ", v.shape)
#         print("q.transpose(-2, -1).shape = ", q.transpose(-2, -1).shape)
#         print("(q.transpose(-2, -1) @ k).shape = ", (q.transpose(-2, -1) @ k).shape)
#         print("self.scalc = ", self.scale)
#         print("(q.transpose(-2, -1) @ k)*self.scale.shape = ", ((q.transpose(-2, -1) @ k) * self.scale).shape)
#         print("attention_biases_idxs.shape:", self.attention_bias_idxs.shape)
#         print("attention_biases.shape:", self.attention_biases.shape)
#         attn_bias = self.attention_biases[:, self.attention_bias_idxs]
#         print("attn_bias.shape:", attn_bias.shape)
#
#         attn = ((q.transpose(-2, -1) @ k) * self.scale+(self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab))
#         # attn = (q.transpose(-2, -1) @ k) * self.scale
#         attn = attn.softmax(dim=-1)
#         x = (v @ attn.transpose(-2, -1)).reshape(B, -1, H, W)
#         x = self.proj(x)
#         return x
class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
    
    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x
    
    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = torch.nn.functional.pad(conv1_w, [1,1,1,1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1,1,1,1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv

import torch.nn as nn

class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)
        
        self.sks = sks
        self.groups = groups
        self.dim = dim
        
    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w

class LSConv(nn.Module):
    def __init__(self, dim):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska(x, self.lkp(x))) + x

class Block(torch.nn.Module):    
    def __init__(self,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 stage=-1, depth=-1):
        super().__init__()
            
        if depth % 2 == 0:
            self.mixer = RepVGGDW(ed)
            self.se = SqueezeExcite(ed, 0.25)
        else:
            self.se = torch.nn.Identity()
            if stage == 3:
                self.mixer = Residual(Attention(ed, kd, nh, ar, resolution=resolution))
            else:
                self.mixer = LSConv(ed)

        self.ffn = Residual(FFN(ed, int(ed * 2)))

    def forward(self, x):
        return self.ffn(self.se(self.mixer(x)))

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, groups=None):
        """
        LSNet 专用下采样模块（匹配原模型逻辑）
        :param in_channels: 输入通道数（如 64/128/192）
        :param out_channels: 输出通道数（如 128/192/256）
        :param stride: 下采样步长（默认2，对应分辨率减半）
        :param groups: 深度卷积的分组数（默认等于输入通道数，即深度卷积）
        """
        super().__init__()
        # 分组数默认=输入通道数（深度卷积，降低计算量）
        if groups is None:
            groups = in_channels

        # 第一步：深度卷积下采样（步长2，分辨率减半，通道数不变）
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(in_channels)
        )

        # 第二步：1×1卷积升维（通道数从in→out，分辨率不变）
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        """前向传播：深度卷积下采样 → 1×1卷积升维"""
        x = self.depthwise_conv(x)  # 下采样：分辨率减半，通道数不变
        x = self.pointwise_conv(x)  # 升维：通道数提升，分辨率不变
        return x

    @torch.no_grad()
    def fuse(self):
        """可选：融合 BN 层到卷积层（推理加速）"""
        # 融合 depthwise_conv 的 Conv+BN
        c1, bn1 = self.depthwise_conv
        w1 = bn1.weight / (bn1.running_var + bn1.eps).sqrt()
        w1 = c1.weight * w1[:, None, None, None]
        b1 = bn1.bias - bn1.running_mean * bn1.weight / (bn1.running_var + bn1.eps).sqrt()
        fused_depthwise = nn.Conv2d(
            c1.in_channels, c1.out_channels, c1.kernel_size,
            stride=c1.stride, padding=c1.padding, groups=c1.groups, bias=True
        )
        fused_depthwise.weight.data.copy_(w1)
        fused_depthwise.bias.data.copy_(b1)

        # 融合 pointwise_conv 的 Conv+BN
        c2, bn2 = self.pointwise_conv
        w2 = bn2.weight / (bn2.running_var + bn2.eps).sqrt()
        w2 = c2.weight * w2[:, None, None, None]
        b2 = bn2.bias - bn2.running_mean * bn2.weight / (bn2.running_var + bn2.eps).sqrt()
        fused_pointwise = nn.Conv2d(
            c2.in_channels, c2.out_channels, c2.kernel_size,
            stride=c2.stride, padding=c2.padding, bias=True
        )
        fused_pointwise.weight.data.copy_(w2)
        fused_pointwise.bias.data.copy_(b2)

        # 替换为融合后的卷积
        self.depthwise_conv = fused_depthwise
        self.pointwise_conv = fused_pointwise
        return self

class LSBlock(torch.nn.Module):
    """
    i = 0, ed = 64, kd = 16, dpth = 1, nh = 4, ar = 1.0
    i = 1, ed = 128, kd = 16, dpth = 2, nh = 4, ar = 2.0
    i = 2, ed = 192, kd = 16, dpth = 3, nh = 4, ar = 3.0
    i = 3, ed = 256, kd = 16, dpth = 4, nh = 4, ar = 4.0
    """
    def __init__(self, ed, kd, dpth, nh, ar, resolution=14, stage=-1, depth=-1):
        super().__init__()
        self.lsBlock = nn.Sequential()
        for d in range(dpth):
            # print(resolution)
            self.lsBlock.append(Block(ed, kd, nh, ar, resolution, stage=stage, depth=d))
    def forward(self, x):
        x = self.lsBlock(x)
        return x

class Stem(nn.Module):
    def __init__(self, in_chans=3, out_chans=64, downsample_ratio=8):
        """
        LSNet 专用Stem模块（输入处理）
        :param in_chans: 输入图像通道数（默认3，RGB图像）
        :param out_chans: Stem最终输出通道数（默认64，匹配原模型）
        :param downsample_ratio: 总下采样倍率（默认8，对应224→28）
        """
        super().__init__()
        # 原模型中Stem分3次下采样（每次步长2，总倍率2^3=8）
        # 通道变化：3 → 16 → 32 → 64
        mid_chans1 = out_chans // 4  # 16
        mid_chans2 = out_chans // 2  # 32

        # 三阶段卷积+BN+ReLU（每次下采样步长2）
        self.stages = nn.Sequential(
            # 阶段1：3→16，224→112
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=mid_chans1,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(mid_chans1),
            nn.ReLU(),

            # 阶段2：16→32，112→56
            nn.Conv2d(
                in_channels=mid_chans1,
                out_channels=mid_chans2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(mid_chans2),
            nn.ReLU(),

            # 阶段3：32→64，56→28
            nn.Conv2d(
                in_channels=mid_chans2,
                out_channels=out_chans,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_chans)
        )

    def forward(self, x):
        """前向传播：输入图像→三阶段下采样+通道提升"""
        # print("Stem: ", self.stages(x).shape)
        return self.stages(x)

    @torch.no_grad()
    def fuse(self):
        """可选：融合Conv+BN（推理加速）"""
        fused_stages = []
        # 每3个模块（Conv+BN+ReLU）为一组，融合Conv+BN
        for i in range(0, len(self.stages), 3):
            conv = self.stages[i]
            bn = self.stages[i + 1]
            relu = self.stages[i + 2]

            # 融合Conv+BN的权重
            w = bn.weight / (bn.running_var + bn.eps).sqrt()
            w = conv.weight * w[:, None, None, None]
            b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps).sqrt()

            # 构建融合后的Conv（带bias）
            fused_conv = nn.Conv2d(
                conv.in_channels, conv.out_channels, conv.kernel_size,
                stride=conv.stride, padding=conv.padding, bias=True
            )
            fused_conv.weight.data.copy_(w)
            fused_conv.bias.data.copy_(b)

            # 保留ReLU
            fused_stages.extend([fused_conv, relu])

        # 替换为融合后的模块
        self.stages = nn.Sequential(*fused_stages)
        return self




class LSNet(torch.nn.Module):
    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=[64, 128, 192, 256],
                 key_dim=[16, 16, 16, 16],
                 depth=[1, 2, 3, 4],
                 num_heads=[4, 4, 4, 4],
                 distillation=False,):
        super().__init__()

        resolution = img_size
        # stem
        self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, embed_dim[0] // 4, 3, 2, 1), torch.nn.ReLU(),
                                Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1), torch.nn.ReLU(),
                                Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1)
                           )

        resolution = img_size // patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.blocks1 = nn.Sequential()
        self.blocks2 = nn.Sequential()
        self.blocks3 = nn.Sequential()
        self.blocks4 = nn.Sequential()
        blocks = [self.blocks1, self.blocks2, self.blocks3, self.blocks4]
        
        for i, (ed, kd, dpth, nh, ar) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio)):
            print("#"*100)
            print(f"i = {i}, ed = {ed}, kd = {kd}, dpth = {dpth}, nh = {nh}, ar = {ar}")
            for d in range(dpth):
                # print(resolution)
                blocks[i].append(Block(ed, kd, nh, ar, resolution, stage=i, depth=d))
            
            if i != len(depth) - 1:
                # print("走了判定逻辑")
                blk = blocks[i+1] # 给下一层先加上Conv2d_BN
                resolution_ = (resolution - 1) // 2 + 1
                blk.append(Conv2d_BN(embed_dim[i], embed_dim[i], ks=3, stride=2, pad=1, groups=embed_dim[i]))
                blk.append(Conv2d_BN(embed_dim[i], embed_dim[i+1], ks=1, stride=1, pad=0))
                resolution = resolution_
            # print(blocks[i])

        self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.head_dist = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
            
        self.num_classes = num_classes
        self.num_features = embed_dim[-1]

    @torch.jit.ignore # type: ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (4, 4),
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.0.c', 'classifier': ('head.linear', 'head_dist.linear'),
        **kwargs
    }

default_cfgs = dict(
    # lsnet_t = _cfg(hf_hub='jameslahm/lsnet_t'),
    # lsnet_t_distill = _cfg(hf_hub='jameslahm/lsnet_t_distill'),
    # lsnet_s = _cfg(hf_hub='jameslahm/lsnet_s'),
    # lsnet_s_distill = _cfg(hf_hub='jameslahm/lsnet_s_distill'),
    # lsnet_b = _cfg(hf_hub='jameslahm/lsnet_b'),
    # lsnet_b_distill = _cfg(hf_hub='jameslahm/lsnet_b_distill'),
    lsnet_t=_cfg(url=""),
    lsnet_t_distill=_cfg(url=""),
    lsnet_s=_cfg(url=""),
    lsnet_s_distill=_cfg(url=""),
    lsnet_b=_cfg(url=""),
    lsnet_b_distill=_cfg(url=""),
)

def _create_lsnet(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        LSNet,
        variant,
        pretrained,
        default_cfg=default_cfgs[variant],
        **kwargs,
    )
    return model

@register_model
def lsnet_t(num_classes=1000, distillation=False, pretrained=False, **kwargs):
    model = _create_lsnet("lsnet_t" + ("_distill" if distillation else ""),
                  pretrained=pretrained,
                  num_classes=num_classes, 
                  distillation=distillation, 
                  img_size=224,
                  patch_size=8,
                  embed_dim=[64, 128, 256, 384],
                  depth=[0, 2, 8, 10],
                  num_heads=[3, 3, 3, 4],
                  )
    return model

@register_model
def lsnet_s(num_classes=1000, distillation=False, pretrained=False, **kwargs):
    model = _create_lsnet("lsnet_s" + ("_distill" if distillation else ""),
                  pretrained=pretrained,
                  num_classes=num_classes, 
                  distillation=distillation,
                  img_size=224,
                  patch_size=8,
                  embed_dim=[96, 192, 320, 448],
                  depth=[1, 2, 8, 10],
                  num_heads=[3, 3, 3, 4],
                  )
    return model

@register_model
def lsnet_b(num_classes=1000, distillation=False, pretrained=False, **kwargs):
    model = _create_lsnet("lsnet_b" + ("_distill" if distillation else ""),
                  pretrained=pretrained,
                  num_classes=num_classes, 
                  distillation=distillation,
                  img_size=224,
                  patch_size=8,
                  embed_dim=[128, 256, 384, 512],
                  depth=[4, 6, 8, 10],
                  num_heads=[3, 3, 3, 4],
                  )
    return model

@register_model
def lsnet_t_distill(**kwargs):
    kwargs["distillation"] = True
    return lsnet_t(**kwargs)

@register_model
def lsnet_s_distill(**kwargs):
    kwargs["distillation"] = True
    return lsnet_s(**kwargs)

@register_model
def lsnet_b_distill(**kwargs):
    kwargs["distillation"] = True
    return lsnet_b(**kwargs)

# if __name__ == '__main__':
#     ls = LSNet()
#     print(ls)