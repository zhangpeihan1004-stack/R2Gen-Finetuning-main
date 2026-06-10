import torch
import torch.nn as nn
import torchvision.models as models
import timm
#R2GEN BASELINE
# class VisualExtractor(nn.Module):
#     def __init__(self, args):
#         super(VisualExtractor, self).__init__()
#         self.visual_extractor = args.visual_extractor
#         self.pretrained = args.visual_extractor_pretrained
#         model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
#         modules = list(model.children())[:-2]
#         self.model = nn.Sequential(*modules)
#         self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
#
#     def forward(self, images):
#         patch_feats = self.model(images)
#         avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
#         batch_size, feat_size, _, _ = patch_feats.shape
#         patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
#         return patch_feats, avg_feats

#twin-transformer
# class VisualExtractor(nn.Module):
#     def __init__(self, args):
#         super(VisualExtractor, self).__init__()
#         self.args = args
#
#         # 模型 'swin_base_patch4_window7_224' (大模型, 效果好)
#         # 'swin_tiny_patch4_window7_224' (小模型, 速度快, 显存占用低)
#         model_name = 'swin_base_patch4_window7_224'
#         print(f"Loading Visual Encoder: {model_name} ...")
#
#         self.swin_model = timm.create_model(model_name, pretrained=True, num_classes=0)
#
#         # Swin Base 输出维度是 1024，Swin Tiny 输出维度是 768
#         # 而 R2Gen 的 Decoder 通常需要 512 (由 args.d_model 决定)
#         # 所以必须加一个 Linear 层把维度压缩下来
#
#         if 'base' in model_name:
#             swin_out_dim = 1024
#         elif 'tiny' in model_name:
#             swin_out_dim = 768
#         else:
#             swin_out_dim = self.swin_model.num_features
#
#         self.projection = nn.Linear(swin_out_dim, args.d_model)
#
#         self.dropout = nn.Dropout(args.dropout)
#
#     def forward(self, images):
#         # Swin 提取特征
#         # 输出形状通常是 (Batch, H, W, Channels) -> 例如 (Batch, 7, 7, 1024)
#         features = self.swin_model.forward_features(images)
#
#         if features.dim() == 4:
#             # 情况 A: Channels Last (timm Swin 的默认行为)
#             # 形状是 (Batch, H, W, C) -> (B, 7, 7, 1024)
#             if features.shape[-1] == self.swin_model.num_features:
#                 b, h, w, c = features.shape
#                 # 需要 (Batch, Seq_Len, Dim) -> (B, 49, 1024)
#                 features = features.view(b, h * w, c)
#                 # 情况 B: Channels First (传统 CNN 格式)
#             # 形状是 (Batch, C, H, W) -> (B, 1024, 7, 7)
#             else:
#                 b, c, h, w = features.shape
#                 # 变成 (Batch, Dim, Seq_Len) -> permute -> (Batch, Seq_Len, Dim)
#                 features = features.view(b, c, h * w).permute(0, 2, 1)
#
#         # 维度投影 (Projection)
#         # 此时 features 形状必须是 (Batch, 49, 1024)
#         # Linear 层期望输入最后一维是 1024
#         patch_feats = self.projection(features)
#         patch_feats = self.dropout(patch_feats)
#
#         # 计算全局特征 (Global Feature)
#         avg_feats = torch.mean(patch_feats, dim=1)
#
#         return patch_feats, avg_feats
#MFA
class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        # 加载 Swin-Base
        # features_only=True: 返回各个 Stage 的中间特征
        # out_indices=(2, 3): 获取 Stage 3 (14x14) 和 Stage 4 (7x7) 的输出
        self.swin_model = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=False,
            features_only=True,
            out_indices=(2, 3)
        )

        # MFA (Multi-scale Feature Aggregation)
        # Conv2d: (In=512, Out=1024, Kernel=3, Stride=2, Padding=1)
        # 将 14x14 下采样到 7x7，并将通道从 512 升到 1024
        self.proj_stage3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # 最终投影层：将融合后的 1024 维特征映射回 d_model (512)
        self.projection = nn.Linear(1024, args.d_model)
        self.dropout = nn.Dropout(args.drop_prob_lm)

    def forward(self, images):
        # 提取多层特征 (timm 输出格式为 Batch, H, W, C)
        stages = self.swin_model(images)

        # feat_s3: (Batch, 14, 14, 512)
        # feat_s4: (Batch, 7, 7, 1024)
        feat_s3 = stages[0]
        feat_s4 = stages[1]

        if feat_s3.shape[-1] == 512:  # 确认是 Channels Last
            feat_s3 = feat_s3.permute(0, 3, 1, 2)  # (B, 14, 14, 512) -> (B, 512, 14, 14)
            feat_s4 = feat_s4.permute(0, 3, 1, 2)  # (B, 7, 7, 1024) -> (B, 1024, 7, 7)

        #  特征融合 (MFA)
        # 对 Stage 3 进行下采样和升维 -> (B, 1024, 7, 7)
        feat_s3_resized = self.proj_stage3(feat_s3)

        # 残差连接 (Element-wise Sum)
        combined_feats = feat_s4 + feat_s3_resized

        #  展平并转换回 Sequence 格式
        # 目前 shape: (B, 1024, 7, 7)
        b, c, h, w = combined_feats.shape

        # 变成 (B, 1024, 49)
        features = combined_feats.view(b, c, h * w)

        # 变成 (B, 49, 1024) 以适应 Linear 层
        features = features.permute(0, 2, 1)

        #  投影到 Decoder 维度 (512)
        patch_feats = self.projection(features)
        patch_feats = self.dropout(patch_feats)

        #  计算全局特征
        avg_feats = torch.mean(patch_feats, dim=1)

        return patch_feats, avg_feats