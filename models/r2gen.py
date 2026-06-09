import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)

        # ==========================================
        # 🌟 核心修改 1: 增加对比学习专用的文本嵌入层，并赋予安全容量
        # ==========================================
        if hasattr(tokenizer, 'get_vocab_size'):
            base_vocab_size = tokenizer.get_vocab_size()
        elif hasattr(tokenizer, 'token2idx'):
            base_vocab_size = len(tokenizer.token2idx)
        else:
            base_vocab_size = len(tokenizer.idx2word)

        print(f"[DEBUG] 探测到的 tokenizer 基础词表大小: {base_vocab_size}")

        # 加上 100 的安全裕量，防止特殊控制符或 padding (-100) 导致越界
        safe_vocab_size = base_vocab_size + 100
        print(f"[DEBUG] 实际使用的 Embedding 安全词表大小: {safe_vocab_size}")

        self.cl_text_embed = nn.Embedding(safe_vocab_size, args.d_model)

        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train', return_features=False):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])  # 正位片特征
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])  # 侧位片特征

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')

            # ==========================================
            # 🌟 核心修改 2: 提取特征并加入越界保护 (clamp)
            # ==========================================
            if return_features and targets is not None:
                # 1. 图像特征：取正侧位片的平均特征
                image_features = (fc_feats_0 + fc_feats_1) / 2.0

                # 2. 文本特征：防止越界报错 (限制在 0 到 max_id 之间)
                # 如果 targets 里包含 -100 (ignore_index)，clamp 会把它变成 0，从而避免越界崩溃
                safe_targets = torch.clamp(targets, min=0, max=self.cl_text_embed.num_embeddings - 1)

                text_embeds = self.cl_text_embed(safe_targets)
                # 取平均得到全局特征
                text_features = text_embeds.mean(dim=1)

                return output, image_features, text_features

        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train', return_features=False):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')

            # ==========================================
            # 🌟 核心修改 3: MIMIC-CXR 同理进行越界保护
            # ==========================================
            if return_features and targets is not None:
                # mimic_cxr 只有一张图
                image_features = fc_feats

                # 文本特征：防止越界报错
                safe_targets = torch.clamp(targets, min=0, max=self.cl_text_embed.num_embeddings - 1)

                text_embeds = self.cl_text_embed(safe_targets)
                text_features = text_embeds.mean(dim=1)

                return output, image_features, text_features

        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output