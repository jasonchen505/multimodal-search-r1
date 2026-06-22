# MMSearch-R1 针对 8 卡 3090/4090 的复现方案

## 资源限制分析

### 用户硬件配置
- **8 卡 3090**: 24GB 显存，约 35.6 TFLOPS（FP16）
- **8 卡 4090**: 24GB 显存，约 82.6 TFLOPS（FP16）

### 原始配置需求
- **32 × H100**: 80GB 显存，约 989 TFLOPS（FP16）
- **显存差距**: 3.3 倍（80GB vs 24GB）
- **计算能力差距**: 约 12-28 倍（H100 vs 3090/4090）

## 复现策略

### 策略 1：完全复现（推荐度：★★☆☆☆）

**目标**：尽可能接近原始配置进行训练

**配置调整**：
```yaml
# 训练配置
data.train_batch_size: 8  # 从 32 减少到 8
data.max_prompt_length: 2048  # 从 4096 减少到 2048
data.max_response_length: 1024  # 从 2048 减少到 1024
actor_rollout_ref.rollout.response_length_total: 4096  # 从 8192 减少到 4096
actor_rollout_ref.rollout.n: 2  # 从 4 减少到 2
actor_rollout_ref.rollout.max_gen_round: 2  # 从 3 减少到 2

# 显存优化配置
actor_rollout_ref.model.enable_gradient_checkpointing: True
actor_rollout_ref.actor.fsdp_config.param_offload: True
actor_rollout_ref.actor.fsdp_config.optimizer_offload: True
actor_rollout_ref.ref.fsdp_config.param_offload: True
actor_rollout_ref.rollout.gpu_memory_utilization: 0.4  # 从 0.6 减少到 0.4
```

**预期效果**：
- 显存使用：约 20-22GB / GPU
- 训练时间：增加 4-6 倍
- 性能：可能略有下降

**风险**：
- 可能出现 OOM（显存不足）
- 训练不稳定（梯度爆炸/消失）
- 需要频繁调整超参数

### 策略 2：LoRA 微调（推荐度：★★★★☆）

**目标**：使用 LoRA 减少可训练参数，降低显存需求

**配置调整**：
```yaml
# LoRA 配置
actor_rollout_ref.model.lora_rank: 8
actor_rollout_ref.model.lora_alpha: 16
actor_rollout_ref.model.lora_dropout: 0.05

# 训练配置
data.train_batch_size: 16  # 可以适当增加
data.max_prompt_length: 2048
data.max_response_length: 1024
actor_rollout_ref.rollout.response_length_total: 4096
actor_rollout_ref.rollout.n: 2
actor_rollout_ref.rollout.max_gen_round: 2

# 显存优化配置
actor_rollout_ref.model.enable_gradient_checkpointing: True
actor_rollout_ref.actor.fsdp_config.param_offload: True
actor_rollout_ref.actor.fsdp_config.optimizer_offload: True
actor_rollout_ref.ref.fsdp_config.param_offload: True
actor_rollout_ref.rollout.gpu_memory_utilization: 0.4
```

**预期效果**：
- 显存使用：约 16-18GB / GPU
- 训练时间：增加 2-3 倍
- 性能：与全量微调相当或略低

**优势**：
- 显存需求大幅降低
- 训练更稳定
- 可以使用更大的批量大小

### 策略 3：模型蒸馏（推荐度：★★★☆☆）

**目标**：使用更小的模型（3B 或 1.5B）进行训练

**配置调整**：
```yaml
# 模型配置
actor_rollout_ref.model.path: Qwen/Qwen2.5-VL-3B-Instruct  # 或 1.5B

# 训练配置
data.train_batch_size: 32  # 可以使用原始批量大小
data.max_prompt_length: 4096
data.max_response_length: 2048
actor_rollout_ref.rollout.response_length_total: 8192
actor_rollout_ref.rollout.n: 4
actor_rollout_ref.rollout.max_gen_round: 3

# 显存优化配置
actor_rollout_ref.model.enable_gradient_checkpointing: True
actor_rollout_ref.actor.fsdp_config.param_offload: False  # 小模型不需要卸载
actor_rollout_ref.actor.fsdp_config.optimizer_offload: False
actor_rollout_ref.ref.fsdp_config.param_offload: False
actor_rollout_ref.rollout.gpu_memory_utilization: 0.6
```

**预期效果**：
- 显存使用：约 12-15GB / GPU
- 训练时间：与原始配置相当
- 性能：可能下降 10-20%

**优势**：
- 可以使用原始配置的大部分参数
- 训练速度快
- 显存充足

**劣势**：
- 模型能力可能不足
- 需要重新训练奖励模型

## 推荐方案

### 首选方案：策略 2（LoRA 微调）

**理由**：
1. 显存需求适中（16-18GB），适合 24GB 显存
2. 训练稳定性好
3. 性能损失较小
4. 可以逐步调整参数

**实施步骤**：

1. **环境准备**
```bash
# 安装依赖
conda create -n mmsearch_r1 python==3.10 -y
conda activate mmsearch_r1
pip3 install -e ./verl
pip3 install vllm==0.8.2
pip3 install transformers==4.51.0
pip3 install flash-attn==2.7.4.post1
pip3 install wandb
pip3 install peft  # LoRA 支持
```

2. **修改配置文件**
```bash
# 复制并修改配置文件
cp mmsearch_r1/scripts/run_mmsearch_r1_grpo.sh mmsearch_r1/scripts/run_mmsearch_r1_grpo_lora.sh
```

3. **修改训练脚本**
```bash
# 在训练脚本中添加 LoRA 配置
python3 -m mmsearch_r1.trainer.multimodal.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA_PATH \
    data.val_files=$VAL_DATA_PATH \
    data.train_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.image_key=images \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.lora_dropout=0.05 \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_sigmoid_decay_warmup=True \
    actor_rollout_ref.actor.optim.lr_sigmoid_decay_ratio=0.95 \
    actor_rollout_ref.actor.optim.lr_sigmoid_decay_warmup_steps=45 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_multi_turn_response_mask=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
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
    actor_rollout_ref.rollout.search.text_search_limit=2 \
    actor_rollout_ref.rollout.search.parallel_tool_call=True \
    actor_rollout_ref.rollout.search.parallel_tool_call_threads=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WANDB_PROJECT_NAME \
    trainer.experiment_name=$WANDB_EXP_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=30 \
    +trainer.search_penalty=0.1 \
    +trainer.format_penalty=0.1 \
    +trainer.reward_mode="EM" \
    +trainer.val_before_train=True \
    +algorithm.filter_groups.enable=False
```

4. **监控训练**
```bash
# 监控 GPU 使用率
nvidia-smi -l 1

# 监控训练日志
tail -f logs/training.log

# 监控 wandb
wandb sync
```

5. **调整参数**
- 如果 OOM：减少 `train_batch_size` 或 `ppo_micro_batch_size_per_gpu`
- 如果训练不稳定：降低学习率或增加 KL 惩罚
- 如果性能不足：增加 `lora_rank` 或 `train_batch_size`

## 备选方案

### 备选方案 1：策略 3（模型蒸馏）

**适用场景**：
- 需要快速验证想法
- 显存严重不足
- 对性能要求不高

**实施步骤**：
1. 使用 Qwen2.5-VL-3B-Instruct 替代 7B
2. 保持原始配置的大部分参数
3. 适当增加批量大小

### 备选方案 2：策略 1（完全复现）

**适用场景**：
- 对性能要求很高
- 有足够的时间进行调试
- 愿意承担训练不稳定的风险

**实施步骤**：
1. 使用原始配置
2. 启用所有显存优化
3. 减少批量大小和序列长度
4. 密切监控训练状态

## 训练时间估算

### LoRA 微调方案
- **每步训练时间**：约 2-4 分钟（取决于批量大小）
- **总训练步数**：约 1000-2000 步
- **总训练时间**：约 30-100 小时

### 模型蒸馏方案
- **每步训练时间**：约 1-2 分钟
- **总训练步数**：约 1000-2000 步
- **总训练时间**：约 15-50 小时

### 完全复现方案
- **每步训练时间**：约 5-10 分钟
- **总训练步数**：约 1000-2000 步
- **总训练时间**：约 80-300 小时

## 注意事项

### 1. 显存管理
- 启用梯度检查点
- 启用参数和优化器卸载
- 减少 vLLM 的 GPU 内存利用率
- 使用混合精度训练

### 2. 训练稳定性
- 监控梯度范数
- 使用较小的学习率
- 启用 KL 惩罚
- 定期保存检查点

### 3. 性能优化
- 使用 Flash Attention
- 启用序列并行
- 优化数据加载
- 使用高效的优化器

### 4. 调试建议
- 先在小数据集上验证
- 逐步增加批量大小
- 监控训练指标
- 定期进行验证

## 总结

对于 8 卡 3090/4090 的配置，**推荐使用 LoRA 微调方案**。该方案在显存需求、训练稳定性和性能之间取得了较好的平衡。如果显存仍然不足，可以考虑使用更小的模型（3B 或 1.5B）。

**关键配置**：
- LoRA rank: 8
- 批量大小: 16
- 序列长度: 2048 + 1024
- 显存优化: 启用所有卸载选项
- 训练时间: 约 30-100 小时

通过合理的配置调整和优化，可以在 8 卡 3090/4090 上成功复现 MMSearch-R1 的核心功能。
