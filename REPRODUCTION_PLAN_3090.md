# MMSearch-R1 全流程复现计划（8×3090 版本）

## 项目背景

**目标**：在 8 卡 3090 (24GB) 上完整复现 MMSearch-R1 项目的全流程

**原始配置**：32 × H100 (80GB)，总显存 2560GB
**用户配置**：8 × 3090 (24GB)，总显存 192GB
**显存差距**：13.3 倍

**核心挑战**：
1. 显存严重不足（24GB vs 80GB）
2. 计算能力差距（3090 vs H100）
3. 需要大量优化才能运行

---

## 复现阶段总览

```
阶段 1：环境准备（Day 1）
├── 1.1 系统环境
├── 1.2 依赖安装
└── 1.3 验证环境

阶段 2：数据准备（Day 1-2）
├── 2.1 下载数据集
├── 2.2 数据预处理
└── 2.3 数据验证

阶段 3：模型准备（Day 2）
├── 3.1 下载模型
├── 3.2 模型验证
└── 3.3 LoRA 配置

阶段 4：训练配置（Day 2-3）
├── 4.1 基础配置
├── 4.2 显存优化配置
└── 4.3 配置验证

阶段 5：推理验证（Day 3）
├── 5.1 单样本推理
├── 5.2 批量推理
└── 5.3 结果验证

阶段 6：LoRA 微调（Day 3-5）
├── 6.1 小规模测试
├── 6.2 完整训练
└── 6.3 训练监控

阶段 7：评估验证（Day 5-6）
├── 7.1 评估脚本
├── 7.2 结果分析
└── 7.3 性能对比

阶段 8：问题排查与优化（Day 6-7）
├── 8.1 常见问题
├── 8.2 性能优化
└── 8.3 经验总结
```

---

## 阶段 1：环境准备（Day 1）

### 1.1 系统环境

**硬件检查**：
```bash
# 检查 GPU 信息
nvidia-smi

# 检查显存
nvidia-smi --query-gpu=memory.total --format=csv

# 检查系统内存
free -h

# 检查磁盘空间
df -h
```

**预期输出**：
- 8 张 3090，每张 24GB 显存
- 系统内存至少 64GB
- 磁盘空间至少 500GB

### 1.2 依赖安装

**创建环境**：
```bash
# 创建 conda 环境
conda create -n mmsearch_r1 python==3.10 -y
conda activate mmsearch_r1

# 安装 PyTorch（CUDA 11.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
cd /home/chenyizhou/multimodal-search-r1
pip3 install -e ./verl

# 安装 vLLM
pip3 install vllm==0.8.2

# 安装 transformers
pip3 install transformers==4.51.0

# 安装 flash-attn
pip3 install flash-attn==2.7.4.post1

# 安装 wandb
pip3 install wandb

# 安装 peft（LoRA 支持）
pip3 install peft

# 安装其他依赖
pip3 install pandas pyarrow pillow opencv-python
```

**配置 wandb**：
```bash
# 登录 wandb
export WANDB_API_KEY="your_api_key"
wandb login $WANDB_API_KEY
```

### 1.3 验证环境

**验证脚本**：
```python
# test_env.py
import torch
import transformers
import vllm
import peft

print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"vLLM: {vllm.__version__}")
print(f"PEFT: {peft.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Memory: {torch.cuda.get_device_properties(i).total_mem / 1024**3:.1f} GB")
```

**运行验证**：
```bash
python test_env.py
```

---

## 阶段 2：数据准备（Day 1-2）

### 2.1 下载数据集

**下载 FVQA 数据集**：
```bash
# 创建数据目录
mkdir -p /home/chenyizhou/data/fvqa

# 下载数据集（HuggingFace）
pip install huggingface_hub

# 下载训练数据
huggingface-cli download lmms-lab/FVQA --local-dir /home/chenyizhou/data/fvqa

# 或者使用 git lfs
git lfs install
git clone https://huggingface.co/datasets/lmms-lab/FVQA /home/chenyizhou/data/fvqa
```

**数据目录结构**：
```
/home/chenyizhou/data/fvqa/
├── train/
│   ├── fvqa_train.parquet
│   └── ...
├── val/
│   ├── fvqa_val.parquet
│   └── ...
└── test/
    ├── fvqa_test.parquet
    └── ...
```

### 2.2 数据预处理

**检查数据格式**：
```python
# check_data.py
import pandas as pd

# 读取训练数据
train_data = pd.read_parquet('/home/chenyizhou/data/fvqa/train/fvqa_train.parquet')

print(f"训练数据量: {len(train_data)}")
print(f"列名: {train_data.columns.tolist()}")
print(f"示例数据:")
print(train_data.head())

# 检查数据分布
if 'search_type' in train_data.columns:
    print(f"\n搜索类型分布:")
    print(train_data['search_type'].value_counts())
```

### 2.3 数据验证

**验证数据完整性**：
```python
# validate_data.py
import pandas as pd
from PIL import Image
import io

def validate_sample(row):
    """验证单个样本"""
    # 检查必要字段
    assert 'prompt' in row, "Missing prompt"
    assert 'images' in row, "Missing images"
    
    # 检查图像
    if 'images' in row and row['images']:
        for img in row['images']:
            if isinstance(img, dict) and 'bytes' in img:
                Image.open(io.BytesIO(img['bytes']))
    
    return True

# 验证前 100 个样本
for i, row in train_data.head(100).iterrows():
    try:
        validate_sample(row)
    except Exception as e:
        print(f"Sample {i} failed: {e}")

print("数据验证完成！")
```

---

## 阶段 3：模型准备（Day 2）

### 3.1 下载模型

**下载 Qwen2.5-VL-7B-Instruct**：
```bash
# 创建模型目录
mkdir -p /home/chenyizhou/models

# 下载模型
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir /home/chenyizhou/models/Qwen2.5-VL-7B-Instruct

# 或者使用 git lfs
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct /home/chenyizhou/models/Qwen2.5-VL-7B-Instruct
```

### 3.2 模型验证

**验证模型**：
```python
# check_model.py
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

model_path = "/home/chenyizhou/models/Qwen2.5-VL-7B-Instruct"

# 加载模型
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print(f"模型加载成功！")
print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
print(f"模型类型: {type(model).__name__}")
```

### 3.3 LoRA 配置

**创建 LoRA 配置**：
```python
# lora_config.py
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=16,  # LoRA alpha
    lora_dropout=0.05,  # Dropout
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 目标模块
    bias="none",
    task_type="CAUSAL_LM"
)

print(f"LoRA 配置:")
print(f"  Rank: {lora_config.r}")
print(f"  Alpha: {lora_config.lora_alpha}")
print(f"  Dropout: {lora_config.lora_dropout}")
print(f"  Target modules: {lora_config.target_modules}")
```

---

## 阶段 4：训练配置（Day 2-3）

### 4.1 基础配置

**创建训练脚本**：
```bash
# 创建训练脚本
cp mmsearch_r1/scripts/run_mmsearch_r1_grpo.sh mmsearch_r1/scripts/run_mmsearch_r1_grpo_3090.sh
```

**修改训练脚本**：
```bash
#!/bin/bash
# run_mmsearch_r1_grpo_3090.sh

cd /home/chenyizhou/multimodal-search-r1

# 设置环境变量
export TRAIN_DATA_PATH=/home/chenyizhou/data/fvqa/train/fvqa_train.parquet
export VAL_DATA_PATH=/home/chenyizhou/data/fvqa/val/fvqa_val.parquet
export WANDB_PROJECT_NAME=mmsearch_r1_3090
export WANDB_EXP_NAME=grpo_lora_v1

# 启动训练
python3 -m mmsearch_r1.trainer.multimodal.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA_PATH \
    data.val_files=$VAL_DATA_PATH \
    data.train_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.image_key=images \
    data.user_prompt_round_1=mmsearch_r1/prompts/round_1_user_prompt_qwenvl.pkl \
    data.user_prompt_after_image_search=mmsearch_r1/prompts/after_image_search_prompt_qwenvl.pkl \
    data.user_prompt_after_text_search=mmsearch_r1/prompts/after_text_search_prompt_qwenvl.pkl \
    actor_rollout_ref.model.path=/home/chenyizhou/models/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.lora_dropout=0.05 \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_sigmoid_decay_warmup=True \
    actor_rollout_ref.actor.optim.lr_sigmoid_decay_ratio=0.95 \
    actor_rollout_ref.actor.optim.lr_sigmoid_decay_warmup_steps=45 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_multi_turn_response_mask=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_multiturn_mmsearch \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.max_gen_round=2 \
    actor_rollout_ref.rollout.response_length_total=4096 \
    actor_rollout_ref.rollout.search.topk=5 \
    actor_rollout_ref.rollout.search.image_search_limit=1 \
    actor_rollout_ref.rollout.search.text_search_limit=1 \
    actor_rollout_ref.rollout.search.parallel_tool_call=True \
    actor_rollout_ref.rollout.search.parallel_tool_call_threads=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WANDB_PROJECT_NAME \
    trainer.experiment_name=$WANDB_EXP_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=30 \
    +trainer.search_penalty=0.1 \
    +trainer.format_penalty=0.1 \
    +trainer.reward_mode="EM" \
    +trainer.val_before_train=True \
    +algorithm.filter_groups.enable=False
```

### 4.2 显存优化配置

**关键配置说明**：

| 配置项 | 原始值 | 3090 值 | 说明 |
|--------|--------|---------|------|
| `data.train_batch_size` | 32 | 8 | 批量大小 |
| `data.max_prompt_length` | 4096 | 2048 | 提示长度 |
| `data.max_response_length` | 2048 | 1024 | 响应长度 |
| `actor_rollout_ref.rollout.n` | 4 | 2 | Rollout 数量 |
| `actor_rollout_ref.rollout.max_gen_round` | 3 | 2 | 最大轮数 |
| `actor_rollout_ref.rollout.response_length_total` | 8192 | 4096 | 总响应长度 |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | 0.6 | 0.4 | GPU 内存利用率 |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | False | True | 参数卸载 |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | False | True | 优化器卸载 |

### 4.3 配置验证

**验证配置**：
```bash
# 检查配置是否正确
python3 -c "
from omegaconf import OmegaConf
config = OmegaConf.load('mmsearch_r1/trainer/multimodal/config/ppo_trainer.yaml')
print('配置加载成功！')
print(f'train_batch_size: {config.data.train_batch_size}')
print(f'max_prompt_length: {config.data.max_prompt_length}')
print(f'max_response_length: {config.data.max_response_length}')
"
```

---

## 阶段 5：推理验证（Day 3）

### 5.1 单样本推理

**推理脚本**：
```python
# inference_test.py
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

model_path = "/home/chenyizhou/models/Qwen2.5-VL-7B-Instruct"

# 加载模型
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 准备输入
image = Image.open("test_image.jpg")
question = "What is shown in this image?"

# 处理输入
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]
    }
]

text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt")
inputs = inputs.to(model.device)

# 生成响应
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512)

# 解码响应
response = processor.decode(outputs[0], skip_special_tokens=True)
print(f"问题: {question}")
print(f"回答: {response}")
```

### 5.2 批量推理

**批量推理脚本**：
```python
# batch_inference.py
import pandas as pd
from tqdm import tqdm

# 加载数据
data = pd.read_parquet('/home/chenyizhou/data/fvqa/val/fvqa_val.parquet')

# 批量推理
results = []
for i, row in tqdm(data.head(10).iterrows(), total=10):
    try:
        # 推理逻辑
        result = inference(row)
        results.append(result)
    except Exception as e:
        print(f"Sample {i} failed: {e}")

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_parquet('inference_results.parquet')
```

### 5.3 结果验证

**验证推理结果**：
```python
# validate_results.py
import pandas as pd

# 加载结果
results = pd.read_parquet('inference_results.parquet')

# 计算准确率
correct = 0
total = len(results)
for i, row in results.iterrows():
    if row['prediction'] == row['ground_truth']:
        correct += 1

accuracy = correct / total
print(f"准确率: {accuracy:.2%}")
print(f"正确: {correct}/{total}")
```

---

## 阶段 6：LoRA 微调（Day 3-5）

### 6.1 小规模测试

**小规模测试脚本**：
```bash
# 小规模测试（100 个样本）
python3 -m mmsearch_r1.trainer.multimodal.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/chenyizhou/data/fvqa/train/fvqa_train.parquet \
    data.val_files=/home/chenyizhou/data/fvqa/val/fvqa_val.parquet \
    data.train_batch_size=4 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    ... # 其他配置
    trainer.total_epochs=1 \
    trainer.total_training_steps=10
```

### 6.2 完整训练

**启动完整训练**：
```bash
# 启动训练
bash mmsearch_r1/scripts/run_mmsearch_r1_grpo_3090.sh
```

**训练监控**：
```bash
# 监控 GPU 使用率
watch -n 1 nvidia-smi

# 监控训练日志
tail -f logs/training.log

# 监控 wandb
wandb sync
```

### 6.3 训练监控

**关键监控指标**：
1. **奖励（Reward）**：应稳步上升
2. **搜索比率（Search Ratio）**：应先升后降，最终稳定
3. **准确率（Accuracy）**：应稳步上升
4. **梯度范数（Grad Norm）**：应保持稳定，不应爆炸
5. **GPU 显存使用**：应保持在 24GB 以下

**监控脚本**：
```python
# monitor_training.py
import wandb
import time

# 初始化 wandb
wandb.init(project="mmsearch_r1_3090", name="training_monitor")

# 监控循环
while True:
    # 读取训练日志
    metrics = read_training_log()
    
    # 记录指标
    wandb.log(metrics)
    
    # 检查异常
    if metrics['grad_norm'] > 100:
        print("警告：梯度范数过大！")
    
    if metrics['search_ratio'] > 0.9:
        print("警告：搜索比率过高！")
    
    time.sleep(10)
```

---

## 阶段 7：评估验证（Day 5-6）

### 7.1 评估脚本

**评估脚本**：
```bash
# 评估脚本
python3 -m mmsearch_r1.trainer.multimodal.main_ppo \
    algorithm.adv_estimator=grpo \
    data.val_files=/home/chenyizhou/data/fvqa/test/fvqa_test.parquet \
    trainer.val_only=True \
    trainer.val_only_save_dir=/home/chenyizhou/results/eval_results \
    trainer.val_generations_to_log_to_wandb=64
```

### 7.2 结果分析

**分析评估结果**：
```python
# analyze_results.py
import pandas as pd
import json

# 加载评估结果
with open('/home/chenyizhou/results/eval_results/val_result_64.json', 'r') as f:
    results = json.load(f)

# 计算指标
correct = sum(1 for r in results if r['score'] > 0.5)
total = len(results)
accuracy = correct / total

# 计算搜索比率
search_count = sum(1 for r in results if r['search_count'] > 0)
search_ratio = search_count / total

print(f"准确率: {accuracy:.2%}")
print(f"搜索比率: {search_ratio:.2%}")
print(f"平均搜索次数: {sum(r['search_count'] for r in results) / total:.2f}")
```

### 7.3 性能对比

**与基线对比**：
```python
# compare_results.py
import pandas as pd

# 加载结果
our_results = pd.read_parquet('/home/chenyizhou/results/eval_results/our_results.parquet')
baseline_results = pd.read_parquet('/home/chenyizhou/results/eval_results/baseline_results.parquet')

# 计算指标
our_accuracy = our_results['score'].mean()
baseline_accuracy = baseline_results['score'].mean()

our_search_ratio = our_results['search_count'].mean()
baseline_search_ratio = baseline_results['search_count'].mean()

print(f"我们的模型:")
print(f"  准确率: {our_accuracy:.2%}")
print(f"  搜索比率: {our_search_ratio:.2%}")

print(f"\n基线模型:")
print(f"  准确率: {baseline_accuracy:.2%}")
print(f"  搜索比率: {baseline_search_ratio:.2%}")

print(f"\n提升:")
print(f"  准确率: {our_accuracy - baseline_accuracy:+.2%}")
print(f"  搜索比率: {our_search_ratio - baseline_search_ratio:+.2%}")
```

---

## 阶段 8：问题排查与优化（Day 6-7）

### 8.1 常见问题

**问题 1：OOM（显存不足）**
```bash
# 解决方案
1. 减少 train_batch_size
2. 减少 max_prompt_length 和 max_response_length
3. 启用 param_offload 和 optimizer_offload
4. 减少 gpu_memory_utilization
5. 使用 LoRA 微调
```

**问题 2：训练不稳定**
```bash
# 解决方案
1. 降低学习率
2. 增加 KL 系数
3. 使用梯度裁剪
4. 检查数据质量
```

**问题 3：搜索比率过高**
```bash
# 解决方案
1. 增加搜索惩罚
2. 确保数据平衡
3. 增加 search-free 样本
```

**问题 4：搜索工具失败**
```bash
# 解决方案
1. 实现缓存机制
2. 实现重试机制
3. 实现降级策略
```

### 8.2 性能优化

**显存优化**：
```python
# 1. 梯度检查点
model.gradient_checkpointing_enable()

# 2. 参数卸载
fsdp_config.param_offload = True
fsdp_config.optimizer_offload = True

# 3. 混合精度
torch_dtype = torch.bfloat16

# 4. 减少批量大小
train_batch_size = 8  # 从 32 减少到 8
```

**计算优化**：
```python
# 1. 减少序列长度
max_prompt_length = 2048  # 从 4096 减少
max_response_length = 1024  # 从 2048 减少

# 2. 减少 Rollout 数量
rollout.n = 2  # 从 4 减少到 2
max_gen_round = 2  # 从 3 减少到 2

# 3. 使用 LoRA
lora_rank = 8
lora_alpha = 16
```

### 8.3 经验总结

**关键经验**：
1. **显存管理**：梯度检查点 + 参数卸载 + 优化器卸载是必须的
2. **批量大小**：从 32 减少到 8 是必要的
3. **序列长度**：从 4096+2048 减少到 2048+1024 是必要的
4. **Rollout 数量**：从 4 减少到 2 是必要的
5. **LoRA 微调**：是 3090 上唯一可行的方案

**预期结果**：
- 准确率：约 50-55%（比原始配置低 3-5%）
- 搜索比率：约 60-70%（与原始配置相当）
- 训练时间：约 50-100 小时

---

## 时间安排

| 阶段 | 任务 | 时间 | 产出 |
|------|------|------|------|
| 阶段 1 | 环境准备 | Day 1 | 环境配置完成 |
| 阶段 2 | 数据准备 | Day 1-2 | 数据集下载完成 |
| 阶段 3 | 模型准备 | Day 2 | 模型下载完成 |
| 阶段 4 | 训练配置 | Day 2-3 | 训练脚本准备完成 |
| 阶段 5 | 推理验证 | Day 3 | 推理验证通过 |
| 阶段 6 | LoRA 微调 | Day 3-5 | 训练完成 |
| 阶段 7 | 评估验证 | Day 5-6 | 评估完成 |
| 阶段 8 | 问题排查 | Day 6-7 | 优化完成 |

**总时间**：约 7 天

---

## 检查清单

### 环境准备
- [ ] GPU 信息检查
- [ ] 系统内存检查
- [ ] 磁盘空间检查
- [ ] conda 环境创建
- [ ] 依赖安装
- [ ] wandb 配置

### 数据准备
- [ ] 数据集下载
- [ ] 数据格式检查
- [ ] 数据完整性验证

### 模型准备
- [ ] 模型下载
- [ ] 模型验证
- [ ] LoRA 配置

### 训练配置
- [ ] 基础配置
- [ ] 显存优化配置
- [ ] 配置验证

### 推理验证
- [ ] 单样本推理
- [ ] 批量推理
- [ ] 结果验证

### LoRA 微调
- [ ] 小规模测试
- [ ] 完整训练
- [ ] 训练监控

### 评估验证
- [ ] 评估脚本
- [ ] 结果分析
- [ ] 性能对比

### 问题排查
- [ ] 常见问题
- [ ] 性能优化
- [ ] 经验总结

---

## 总结

本复现计划针对 8 卡 3090 (24GB) 的配置，通过以下关键优化实现 MMSearch-R1 的完整复现：

1. **显存优化**：梯度检查点 + 参数卸载 + 优化器卸载
2. **批量大小**：从 32 减少到 8
3. **序列长度**：从 4096+2048 减少到 2048+1024
4. **Rollout 数量**：从 4 减少到 2
5. **LoRA 微调**：使用 LoRA 减少可训练参数

通过这些优化，可以在 3090 上实现约 50-55% 的准确率，比原始配置低 3-5%，但搜索比率与原始配置相当（60-70%）。

**关键成功因素**：
1. 正确配置显存优化参数
2. 确保数据平衡（search-required + search-free）
3. 监控训练指标，及时调整超参数
4. 实现搜索工具的缓存和重试机制
