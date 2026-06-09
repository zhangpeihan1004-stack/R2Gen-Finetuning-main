import torch
import torch.nn as nn
import torch.nn.functional as F


# class LanguageModelCriterion(nn.Module):
#     def __init__(self):
#         super(LanguageModelCriterion, self).__init__()
#
#     def forward(self, input, target, mask, image_features=None, text_features=None, lambda_cl=0.1, gamma=2.0, **kwargs):
#
#         if target.size(1) > input.size(1):
#             target = target[:, 1:input.size(1) + 1]
#             mask = mask[:, 1:input.size(1) + 1]
#         else:
#             target = target[:, :input.size(1)]
#             mask = mask[:, :input.size(1)]
#
#         vocab_size = input.size(-1)
#
#         target_safe = torch.clamp(target, min=0, max=vocab_size - 1)
#
#         # 取出正确单词对应的概率
#         log_pt = input.gather(2, target_safe.long().unsqueeze(2)).squeeze(2)
#
#         # 计算传统的文本交叉熵 Loss
#         loss_lm = -1.0 * log_pt * mask
#         loss_lm = torch.sum(loss_lm) / torch.sum(mask)
#
#         # 计算对比损失 (Contrastive Loss)
#         loss_cl = 0
#         if image_features is not None and text_features is not None:
#             current_batch_size = image_features.size(0)
#             labels = torch.arange(current_batch_size).to(image_features.device)
#
#             logits_img = image_features @ text_features.T
#             logits_txt = text_features @ image_features.T
#
#             loss_cl = (F.cross_entropy(logits_img, labels) + F.cross_entropy(logits_txt, labels)) / 2.0
#         # 返回总 Loss
#         return loss_lm + lambda_cl * loss_cl

#Facal Loss
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask, gamma=2.0, **kwargs):
        # 截断 target 和 mask 以匹配 input 长度
        if target.size(1) > input.size(1):
            target = target[:, 1:input.size(1) + 1]
            mask = mask[:, 1:input.size(1) + 1]
        else:
            target = target[:, :input.size(1)]
            mask = mask[:, :input.size(1)]

        vocab_size = input.size(-1)
        target_safe = torch.clamp(target, min=0, max=vocab_size - 1)

        # 1. 取得 log(p_t)
        log_pt = input.gather(2, target_safe.long().unsqueeze(2)).squeeze(2)

        # 2. 取得真实的概率 p_t (将 log_pt 用 exp 还原)
        pt = torch.exp(log_pt)

        # 3. 计算 Focal Loss 的核心调制系数: (1 - p_t)^gamma
        focal_weight = (1 - pt) ** gamma

        # 4. 完整的 Focal Loss 计算
        loss_lm = -1.0 * focal_weight * log_pt * mask

        # 求平均
        loss_lm = torch.sum(loss_lm) / torch.sum(mask)

        return loss_lm
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        # L2 归一化
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # 计算相似度矩阵 (B x B)
        logits_per_image = torch.matmul(image_features, text_features.t()) / self.temperature
        logits_per_text = logits_per_image.t()

        # 生成对角线标签 (0, 1, 2... B-1)
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=image_features.device)

        # 双向交叉熵
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        return (loss_i2t + loss_t2i) / 2.0

# def compute_loss(output, reports_ids, reports_masks, image_features=None, text_features=None, lambda_cl=0.1):
#
#     # 计算原本的文本生成 Focal Loss
#     lm_criterion = LanguageModelCriterion()
#     lm_loss = lm_criterion(output, reports_ids[:, 1:], reports_masks[:, 1:], gamma=2.0)
#
#     # 如果传入了图文特征，则额外计算对比 Loss
#     if image_features is not None and text_features is not None:
#         cl_criterion = ContrastiveLoss(temperature=0.07).to(image_features.device)
#         cl_loss = cl_criterion(image_features, text_features)
#
#         # 将两个 Loss 加权求和
#         total_loss = lm_loss + lambda_cl * cl_loss
#         return total_loss
#     else:
#         # 如果没传特征，就只返回原本的文本 Loss
#         return lm_loss
def compute_loss(output, reports_ids, reports_masks, image_features=None, text_features=None,
                 lambda_cl=0.1, focal_gamma=2.0, cl_temperature=0.07):

    # 将外部传来的 focal_gamma 传给 LanguageModelCriterion
    lm_criterion = LanguageModelCriterion()
    lm_loss = lm_criterion(output, reports_ids[:, 1:], reports_masks[:, 1:], gamma=focal_gamma)

    if image_features is not None and text_features is not None:
        # 将外部传来的 cl_temperature 传给 ContrastiveLoss
        cl_criterion = ContrastiveLoss(temperature=cl_temperature).to(image_features.device)
        cl_loss = cl_criterion(image_features, text_features)

        total_loss = lm_loss + lambda_cl * cl_loss
        return total_loss
    else:
        return lm_loss