from ultralytics.nn.modules import lsnet
import torch


class LSTest(torch.nn.Module):
    """
    i = 0, ed = 64, kd = 16, dpth = 1, nh = 4, ar = 1.0
    i = 1, ed = 128, kd = 16, dpth = 2, nh = 4, ar = 2.0
    i = 2, ed = 192, kd = 16, dpth = 3, nh = 4, ar = 3.0
    i = 3, ed = 256, kd = 16, dpth = 4, nh = 4, ar = 4.0

    i = 0, ed = 128, kd = 16, dpth = 4, nh = 3, ar = 1.0
    i = 1, ed = 256, kd = 16, dpth = 6, nh = 3, ar = 2.0
    i = 2, ed = 384, kd = 16, dpth = 8, nh = 3, ar = 3.0
    i = 3, ed = 512, kd = 16, dpth = 10, nh = 4, ar = 4.0
    """

    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=[64, 128, 192, 256],
                 key_dim=[16, 16, 16, 16],
                 depth=[1, 2, 3, 4],
                 num_heads=[4, 4, 4, 4],
                 distillation=False, ):
        super().__init__()
        resolution = img_size // patch_size
        self.stem = lsnet.Stem(3, 128)
        self.lsBlock1 = lsnet.LSBlock(ed=128, kd=16, dpth=1, nh=4, ar=1.0, stage=0, resolution=resolution)
        resolution = (resolution - 1) // 2 + 1
        self.downsample1 = lsnet.Downsample(128, 256, stride=2)
        self.lsBlock2 = lsnet.LSBlock(ed=256, kd=16, dpth=2, nh=4, ar=2.0, stage=1, resolution=resolution)
        resolution = (resolution - 1) // 2 + 1
        self.downsample2 = lsnet.Downsample(256, 384, stride=2)
        self.lsBlock3 = lsnet.LSBlock(ed=384, kd=16, dpth=3, nh=4, ar=3.0, stage=2, resolution=resolution)
        resolution = (resolution - 1) // 2 + 1
        self.downsample3 = lsnet.Downsample(384, 512, stride=2)
        self.mas = lsnet.LSBlock(ed=512, kd=16, dpth=4, nh=4, ar=4.0, stage=3, resolution=resolution)
        self.to(device)
    def forward(self, x):
        x = self.stem(x)
        x = self.lsBlock1(x)
        x = self.downsample1(x)
        x = self.lsBlock2(x)
        x = self.downsample2(x)
        x = self.lsBlock3(x)
        x = self.downsample3(x)
        print(x.shape)
        x = self.mas(x)
        return x

if __name__ == '__main__':
    # q = torch.randn(1, 4, 16, 16)  # [B, num_heads, N, key_dim]
    # k = torch.randn(1, 4, 16, 16)  # [B, num_heads, key_dim, N]
    #
    # # 方法1：直接调整k的维度（推荐，符合注意力机制逻辑）
    # attn = q @ k  # 合法：[1,4,16,16] @ [1,4,16,16] → [1,4,16,16]
    # print("注意力权重维度：", attn.shape)  # torch.Size([1, 4, 16, 16])
    #
    # # 方法2：若k维度是[B, num_heads, N, key_dim]，需先转置k
    # k_trans = k.transpose(-2, -1)  # [1,4,16,16] → [1,4,16,16]（仅演示，实际根据真实维度调整）
    # attn2 = q @ k_trans
    # print("attn2维度：", attn2.shape)  # torch.Size([1, 4, 16, 16])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = lsnet.LSNet()
    # model = LSTest()
    model = lsnet.lsnet_b(img_size=640, patch_size=8)

    model.to(device)
    input = torch.randn(1, 3, 640, 640).to(device)
    # input = torch.randn(1, 256, 14, 14).to(device)
    print(model)
    output = model(input)
    print(output.shape)
