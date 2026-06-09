#!/bin/bash
#SBATCH --job-name=ZPH_swin-trans
#SBATCH --partition=gpu-a100        # 👉 修改 1：指定 A100 队列（具体名字视你们集群而定，通常是 gpu-a100 或 a100）
#SBATCH --gres=gpu:1                # 👉 修改 2：申请 1 张 GPU（也就是单卡）
#SBATCH --nodes=1                   
#SBATCH --cpus-per-task=8           # CPU 核心数保持 8 个一般够用
#SBATCH --mem=32G                   # 👉 修改 3：A100 算得快，吃数据也快，建议把系统内存稍微调大点（比如 64G），防止数据处理成瓶颈
#SBATCH --time=8:00:00             
#SBATCH --output=train_log_%j.txt   

# ================= 环境加载与运行 (保持不变) =================
# 1. 只需要加载基础 conda，不在这里搞容易失败的 source activate
module load miniconda/24.11.1

# 2. 切换到工作目录
cd /scr/user/zhangpeihan1004/swin-trans/R2Gen-Finetuning-main

# 3. 关键绝杀！直接用你专属环境里的 Python 绝对路径去运行代码
/home/user/zhangpeihan1004/.conda/envs/my_project_py310/bin/python main.py
