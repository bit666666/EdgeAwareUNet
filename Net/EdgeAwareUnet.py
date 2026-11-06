import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
"""--------------------------------------module--------------------------------------"""
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel,kernel_size=3,stride=1,padding=1):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)
class EdgeAttentionModule(nn.Module):
    """
    边缘注意力模块：增强对小目标边缘和细节的关注
    特点：
    1. 捕获多尺度边缘特征
    2. 生成精细的边缘注意力图
    3. 与主特征融合，增强小目标特征表达
    """

    def __init__(self, channel, reduction=4):
        super(EdgeAttentionModule, self).__init__()
        # 边缘检测卷积 - 使用不同的边缘检测核
        self.edge_conv1 = nn.Conv2d(channel, channel // reduction, kernel_size=3, stride=1, padding=1)
        self.edge_conv2 = nn.Conv2d(channel, channel // reduction, kernel_size=3, stride=1, padding=2, dilation=2)

        # Sobel算子用于边缘检测
        self.register_buffer('sobel_x', torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ]).view(1, 1, 3, 3).repeat(channel // reduction, 1, 1, 1))

        self.register_buffer('sobel_y', torch.tensor([
            [-1., -2., -1.],
            [0., 0., 0.],
            [1., 2., 1.]
        ]).view(1, 1, 3, 3).repeat(channel // reduction, 1, 1, 1))

        # 注意力生成
        self.attention = nn.Sequential(
            nn.Conv2d(channel // reduction * 2, channel // reduction, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        # 特征细化
        self.refine_conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 基础边缘特征提取
        b,c,h,w=x.size()
        edge1 = self.edge_conv1(x)
        edge2 = self.edge_conv2(x)

        # 使用Sobel算子增强边缘检测
        edge_x = F.conv2d(edge1, self.sobel_x, groups=edge1.size(1), padding=1)
        edge_y = F.conv2d(edge1, self.sobel_y, groups=edge1.size(1), padding=1)
        edge_detail = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)  # 边缘强度

        edge_feat = torch.cat([edge_detail, edge2], dim=1)

        # 生成注意力图
        attn_map = self.attention(edge_feat)
        # attn_map_numpy = attn_map.cpu().numpy()
        # np.save(f'./attention_numpy/attn_map_{c}.npy', attn_map_numpy)
        # 应用注意力到原始特征
        attn_feat = x * attn_map

        # 特征细化并与原始特征残差连接
        refined_feat = self.refine_conv(attn_feat) + x

        return refined_feat
class EdgeEncoder(nn.Module):
    #Edge Attention
    def __init__(self, in_channel, out_channel, r=2):
        super(EdgeEncoder, self).__init__()
        self.main_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
        )
        self.residual_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)

        #Edge Attention
        self.edge_attention = EdgeAttentionModule(out_channel,r)
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        main_out = self.main_conv(x)
        residual = self.residual_conv(x)
        main_out = main_out + residual

        # 应用边缘注意力
        attn_out = self.edge_attention(main_out)

        return self.final_relu(attn_out)
class DDPU(nn.Module):
    #Dual-path Dynamic Pixel Upsampling / Dual-path Gated Dynamic Upsampling
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DDPU, self).__init__()

        if mid_channels is None:
            mid_channels = max(in_channels // 2, 16)

        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        #interp_upsample
        self.interp_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        )

        #deconv_upsample
        self.deconv_upsample = nn.ConvTranspose2d(
            mid_channels, out_channels, kernel_size=2, stride=2
        )

        #gate
        self.gate_network = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.residual = None
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        feat = self.feature_conv(x)
        deconv_out = self.deconv_upsample(feat)
        interp_out = self.interp_upsample(feat)

        upsampled_feat = F.interpolate(feat, size=deconv_out.shape[2:], mode='bilinear', align_corners=False)
        gate = self.gate_network(upsampled_feat)
        # fusion
        fused_output = gate * deconv_out + (1 - gate) * interp_out

        if self.residual is not None:
            residual = self.residual(residual)
        residual = F.interpolate(residual, size=deconv_out.shape[2:], mode='bilinear', align_corners=False)

        return fused_output + residual
class EEM(nn.Module):
    #Edge Enhancement Module
    def __init__(self, in_channels):
        super(EEM, self).__init__()
        edge_feat_channels=in_channels*2
        self.edge_feat = nn.Sequential(
            nn.Conv2d(in_channels, edge_feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(edge_feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(edge_feat_channels, edge_feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(edge_feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(edge_feat_channels, 1, kernel_size=1),
            nn.Sigmoid()  #[B,1,H,W]
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.enhance_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        edge_map = self.edge_feat(x)  # [B,1,H,W]

        # 计算空间注意力（基于输入x和边缘图）
        avg_feat = torch.mean(x, dim=1, keepdim=True)  # [B,1,H,W]
        spatial_input = torch.cat([avg_feat, edge_map], dim=1)
        spatial_att = self.spatial_attention(spatial_input)  # [B,1,H,W]

        enhanced = self.enhance_conv(x * spatial_att * edge_map)
        out = x + enhanced

        return out

"""---------------------------------------net---------------------------------------"""
class baseline_EdgeEncoder(nn.Module):
    def __init__(self, classes=3, c=[16, 32, 64, 128, 256]):
        super(baseline_EdgeEncoder, self).__init__()
        self.pool = nn.MaxPool2d(2)

        self.encoderConv1 = EdgeEncoder(3, c[0],2)
        self.encoderConv2 = EdgeEncoder(c[0], c[1],2)
        self.encoderConv3 = EdgeEncoder(c[1], c[2],2)
        self.encoderConv4 = EdgeEncoder(c[2], c[3],2)

        #bottleneck
        self.bottleneck = ConvBlock(c[3], c[4])

        self.decoderConv1 = ConvBlock(c[4] + c[3], c[3])
        self.decoderConv2 = ConvBlock(c[3] + c[2], c[2])
        self.decoderConv3 = ConvBlock(c[2] + c[1], c[1])
        self.decoderConv4 = ConvBlock(c[1] + c[0], c[0])
        # 输出
        self.out = nn.Conv2d(c[0], classes, kernel_size=1)

    def forward(self, x):
        en_x1 = self.encoderConv1(x)
        down_x1 = self.pool(en_x1)
        en_x2 = self.encoderConv2(down_x1)
        down_x2 = self.pool(en_x2)
        en_x3 = self.encoderConv3(down_x2)
        down_x3 = self.pool(en_x3)
        en_x4 = self.encoderConv4(down_x3)
        down_x4 = self.pool(en_x4)
        bottleneck = self.bottleneck(down_x4)
        de_x1 = F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True)
        merge1 = torch.cat([en_x4, de_x1], dim=1)
        de_x1 = self.decoderConv1(merge1)

        de_x2 = F.interpolate(de_x1, scale_factor=2, mode='bilinear', align_corners=True)
        merge2 = torch.cat([en_x3, de_x2], dim=1)
        de_x2 = self.decoderConv2(merge2)

        de_x3 = F.interpolate(de_x2, scale_factor=2, mode='bilinear', align_corners=True)
        merge3 = torch.cat([en_x2, de_x3], dim=1)
        de_x3 = self.decoderConv3(merge3)

        de_x4 = F.interpolate(de_x3, scale_factor=2, mode='bilinear', align_corners=True)
        merge4 = torch.cat([en_x1, de_x4], dim=1)
        de_x4 = self.decoderConv4(merge4)
        return self.out(de_x4)
class baseline_EdgeEncoder_DDPU(nn.Module):
    def __init__(self, classes=3, c=[16, 32, 64, 128, 256]):
        super(baseline_EdgeEncoder_DDPU, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.encoderConv1 = EdgeEncoder(3, c[0], 2)
        self.encoderConv2 = EdgeEncoder(c[0], c[1], 2)
        self.encoderConv3 = EdgeEncoder(c[1], c[2], 2)
        self.encoderConv4 = EdgeEncoder(c[2], c[3], 2)
        #bottleneck
        self.bottleneck = ConvBlock(c[3], c[4])
        self.ddpu1 = DDPU(c[4], c[4])
        self.decoderConv1 = ConvBlock(c[4] + c[3], c[3])
        self.ddpu2 = DDPU(c[3], c[3])
        self.decoderConv2 = ConvBlock(c[3] + c[2], c[2])
        self.ddpu3 = DDPU(c[2], c[2])
        self.decoderConv3 = ConvBlock(c[2] + c[1], c[1])
        self.ddpu4 = DDPU(c[1], c[1])
        self.decoderConv4 = ConvBlock(c[1] + c[0], c[0])
        self.out = nn.Conv2d(c[0], classes, kernel_size=1)

    def forward(self, x):
        en_x1 = self.encoderConv1(x)
        down_x1 = self.pool(en_x1)
        en_x2 = self.encoderConv2(down_x1)
        down_x2 = self.pool(en_x2)
        en_x3 = self.encoderConv3(down_x2)
        down_x3 = self.pool(en_x3)
        en_x4 = self.encoderConv4(down_x3)
        down_x4 = self.pool(en_x4)
        #bottleneck
        bottleneck = self.bottleneck(down_x4)
        de_x1 = self.ddpu1(bottleneck)
        merge1 = torch.cat([en_x4, de_x1], dim=1)
        de_x1 = self.decoderConv1(merge1)

        de_x2 = self.ddpu2(de_x1)
        merge2 = torch.cat([en_x3, de_x2], dim=1)
        de_x2 = self.decoderConv2(merge2)

        de_x3 = self.ddpu3(de_x2)
        merge3 = torch.cat([en_x2, de_x3], dim=1)
        de_x3 = self.decoderConv3(merge3)

        de_x4 = self.ddpu4(de_x3)
        merge4 = torch.cat([en_x1, de_x4], dim=1)
        de_x4 = self.decoderConv4(merge4)
        return self.out(de_x4)
"""---------------------------------------------------------------------------------"""
class EdgeAwareUnet(nn.Module):
    def __init__(self, classes=3, c=[16, 32, 64, 128, 256]):
        super(EdgeAwareUnet, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.encoderConv1 = EdgeEncoder(3, c[0], 2)
        self.encoderConv2 = EdgeEncoder(c[0], c[1], 2)
        self.encoderConv3 = EdgeEncoder(c[1], c[2], 2)
        self.encoderConv4 = EdgeEncoder(c[2], c[3], 2)

        self.bottleneck = ConvBlock(c[3], c[4])

        self.ddpu1 = DDPU(c[4], c[4])
        self.eem1 = EEM(c[3])
        self.decoderConv1 = ConvBlock(c[4] + c[3], c[3])

        self.ddpu2 = DDPU(c[3], c[3])
        self.eem2 = EEM(c[2])
        self.decoderConv2 = ConvBlock(c[3] + c[2], c[2])

        self.ddpu3 = DDPU(c[2], c[2])
        self.eem3 = EEM(c[1])
        self.decoderConv3 = ConvBlock(c[2] + c[1], c[1])

        self.ddpu4 = DDPU(c[1], c[1])
        self.eem4 = EEM(c[0])
        self.decoderConv4 = ConvBlock(c[1] + c[0], c[0])

        self.out = nn.Conv2d(c[0], classes, kernel_size=1)
        # 深度监督
        self.sup1 = self._make_supervision_layer(c[3], classes)
        self.sup2 = self._make_supervision_layer(c[2], classes)
        self.sup3 = self._make_supervision_layer(c[1], classes)
        self.sup4 = self._make_supervision_layer(c[0], classes)

    def _make_supervision_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        en_x1 = self.encoderConv1(x)
        down_x1 = self.pool(en_x1)
        en_x2 = self.encoderConv2(down_x1)
        down_x2 = self.pool(en_x2)
        en_x3 = self.encoderConv3(down_x2)
        down_x3 = self.pool(en_x3)
        en_x4 = self.encoderConv4(down_x3)
        down_x4 = self.pool(en_x4)

        bottleneck = self.bottleneck(down_x4)

        de_x1 = self.ddpu1(bottleneck)
        cat_1 = self.eem1(en_x4)
        merge1 = torch.cat([cat_1, de_x1], dim=1)
        de_x1 = self.decoderConv1(merge1)

        de_x2 = self.ddpu2(de_x1)
        cat_2 = self.eem2(en_x3)
        merge2 = torch.cat([cat_2, de_x2], dim=1)
        de_x2 = self.decoderConv2(merge2)

        de_x3 = self.ddpu3(de_x2)
        cat_3 = self.eem3(en_x2)
        merge3 = torch.cat([cat_3, de_x3], dim=1)
        de_x3 = self.decoderConv3(merge3)

        de_x4 = self.ddpu4(de_x3)
        cat_4 = self.eem4(en_x1)
        merge4 = torch.cat([cat_4, de_x4], dim=1)
        de_x4 = self.decoderConv4(merge4)

        out = self.out(de_x4)

        if self.training:
            sup1 = self.sup1(F.interpolate(cat_1, size=x.shape[2:], mode='bilinear', align_corners=False))
            sup2 = self.sup2(F.interpolate(cat_2, size=x.shape[2:], mode='bilinear', align_corners=False))
            sup3 = self.sup3(F.interpolate(cat_3, size=x.shape[2:], mode='bilinear', align_corners=False))
            sup4 = self.sup4(F.interpolate(cat_4, size=x.shape[2:], mode='bilinear', align_corners=False))
            return [out, sup4, sup3, sup2, sup1]
        else:
            return out
"""-------------------------------------------------------------------"""

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化模型
    model = EdgeAwareUnet(2).to(device)
    model.eval()  # 推理时应使用eval模式

    # 创建输入张量
    input_image = torch.randn(1, 3, 256, 512).to(device)

    # 预热模型（排除首次推理的额外开销）
    with torch.no_grad():
        for _ in range(10):
            output = model(input_image)

    # 测量推理时间
    with torch.no_grad():
        start_time = time.time()
        # 多次推理取平均
        num_runs = 100
        for _ in range(num_runs):
            output = model(input_image)
        end_time = time.time()

    # 计算FPS
    total_time = end_time - start_time
    avg_time_per_image = total_time / num_runs
    fps = 1 / avg_time_per_image

    # 计算FLOPs和参数量
    flops, params = profile(model, inputs=(input_image,))

    # 输出结果
    print(f"输入尺寸: {input_image.shape}")
    print(f"输出尺寸: {output[0].shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"计算量: {flops / 1e9:.2f} GFLOPs")
    print(f"平均推理时间: {avg_time_per_image * 1000:.2f} ms")
    print(f"FPS: {fps:.2f}")