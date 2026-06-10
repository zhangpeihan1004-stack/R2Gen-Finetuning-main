import os

# 设置镜像与调试环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
from PIL import Image
from torchvision import transforms
import modules.tokenizers
from models.r2gen import R2GenModel
import urllib.request

# =====================================================================
# 1. 终极参数配置
# =====================================================================
class Args:
    ann_path = 'data/iu_xray/annotation.json'
    image_dir = 'data/iu_xray/images/'
    vocab_path = 'data/iu_xray/vocab8.pkl'  # 自动对齐你本地的 vocab8.pkl

    dataset_name = 'iu_xray'
    max_seq_length = 100
    threshold = 2
    num_workers = 0
    batch_size = 1

    # 视觉特征提取器
    visual_extractor = 'swin_base_mfa'
    visual_extractor_pretrained = True

    # Transformer 结构参数
    d_model = 512
    d_ff = 512
    d_vf = 512
    num_heads = 8
    num_layers = 3
    dropout = 0.5
    logit_layers = 1
    bos_idx, eos_idx, pad_idx = 0, 0, 0
    use_bn = 0
    drop_prob_lm = 0.5

    # 关系内存模块
    rm_num_slots = 3
    rm_num_heads = 8
    rm_d_model = 512

    # 解码策略
    sample_method = 'beam_search'
    beam_size = 3
    temperature = 1.0
    sample_n = 1
    group_size = 1
    output_logsoftmax = 1
    decoding_constraint = 0
    block_trigrams = 1

    # Loss 占位符
    lambda_cl = 0.1
    focal_gamma = 2.0
    cl_temperature = 0.07
    seed = 9233
    n_gpu = 1
    epochs = 1
    resume = None


# =====================================================================
# 2. 初始化环境与模型
# =====================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("⏳ 正在组装 Swin-MFA + Transformer 模型结构...")
args = Args()
tokenizer = modules.tokenizers.Tokenizer(args)
model = R2GenModel(args, tokenizer).to(device)

# =====================================================================
# 🌟 智能权重托管与自动下载
# =====================================================================
ckpt_path = 'results/iu_xray/checkpoint_epoch_4_Best.pth'

if not os.path.exists(ckpt_path):
    print("侦测到云端未携带权重文件，正在启动动态托管下载...")
    # 创建本地存放权重的多级文件夹
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    # 🔗 这里替换成你的直链下载 URL（务必是点击就能直接弹出下载保存框的链接）
    # 示例：你可以把权重丢到 Hugging Face 的某个公开 Repo 里的 resolve 链接
    download_url = "https://huggingface.co/labmoby/r2gen_mfa/resolve/main/checkpoint_epoch_4_Best.pth"

    try:
        print(f"📥 正在从远程服务器下载模型权重至: {ckpt_path} ...")
        urllib.request.urlretrieve(download_url, ckpt_path)
        print("✨ 权重文件成功下载并安全落盘！")
    except Exception as e:
        print(f"❌ 下载失败，请检查链接是否有效。错误信息: {e}")

# 正常读取落盘后的权重
print(f"📥 正在读取微调权重存档: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.eval()
print("✅ 模型、词表、微调参数全状态加载成功！应用正式就绪！\n")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


# =====================================================================
# 3. 智能兼容函数：支持直接输入文件夹（读两张图）或单张图片文件
# =====================================================================
def predict_patient_images(path):
    if not os.path.exists(path):
        return f"❌ 错误：找不到路径！请检查路径是否正确: {path}"

    # 🌟 情况 A：用户传入的是一个文件夹（包含正位和侧位两张图）
    if os.path.isdir(path):
        # 扫描文件夹下所有的图片格式文件
        valid_exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
        img_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(valid_exts)]
        img_files.sort()  # 排序，确保读取的稳定性

        if len(img_files) == 0:
            return f"❌ 错误：目标文件夹中没有任何图片文件！"

        if len(img_files) >= 2:
            print(f"📸 智能识别成功！检测到病人目录，已自动提取双视角 X 光片进行联合诊断：")
            print(f"   正面视角 ➡️ {os.path.basename(img_files[0])}")
            print(f"   侧面视角 ➡️ {os.path.basename(img_files[1])}")

            img1 = Image.open(img_files[0]).convert('RGB')
            img2 = Image.open(img_files[1]).convert('RGB')
            img1_t = transform(img1).to(device)
            img2_t = transform(img2).to(device)
        else:
            print(f"⚠️ 提示：文件夹内只有 1 张图，自动通过镜像复制凑齐双视角。")
            img1 = Image.open(img_files[0]).convert('RGB')
            img1_t = transform(img1).to(device)
            img2_t = img1_t

    # 🌟 情况 B：用户传入的是某一张特定图片的文件路径
    else:
        print(f"📸 检测到单张图片输入，自动复制该图假装双视角格式进行诊断。")
        img1 = Image.open(path).convert('RGB')
        img1_t = transform(img1).to(device)
        img2_t = img1_t

    # 完美的组装成 [1, 2, C, H, W] 模型所需的五维输入
    image_tensor = torch.stack([img1_t, img2_t], dim=0).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor, mode='sample')
        report = model.tokenizer.decode_batch(output.cpu().numpy())[0]

    return report


# =====================================================================
# 4. 代码测试入口
# =====================================================================
if __name__ == '__main__':
    # 指向你的病人文件夹路径（现在可以直接写文件夹了！）
    my_image_path = 'data/iu_xray/images/CXR1491_IM-0317'

    print(f"🔍 正在分析诊断目标: {my_image_path}")
    print("=" * 60)

    # 联合诊断
    report_result = predict_patient_images(my_image_path)

    # 打印最终诊断结果
    print(f"\n🩺 放射科医生（Swin-MFA 双视角全状态）最终诊断报告：\n\n{report_result}")
    print("=" * 60)