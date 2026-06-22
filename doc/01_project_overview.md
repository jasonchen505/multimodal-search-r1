# MMSearch-R1 项目概述

## 项目简介

MMSearch-R1 是一个端到端的强化学习（RL）框架，旨在使大型多模态模型（LMMs）能够执行按需的、多轮搜索，使用真实的多模态搜索工具。该项目由字节跳动和南洋理工大学联合开发。

## 核心特性

### 1. 多模态搜索能力
- **图像搜索**：基于 SerpAPI，模型可以提交图像 URL，获取视觉相关的网页信息
- **文本搜索**：结合 SerpAPI、JINA Reader 和 Qwen3-32B 进行文本搜索和摘要

### 2. 多轮对话训练
- 支持最多 3 轮对话的 rollout
- 每轮可以调用搜索工具或直接回答
- 使用 GRPO（Group Relative Policy Optimization）算法

### 3. 奖励机制
- **格式奖励**：检查响应是否符合规定的格式
- **正确性奖励**：使用精确匹配（EM）或子串匹配（SubEM）
- **搜索惩罚**：对过度使用搜索工具进行惩罚

## 技术栈

### 基础模型
- **Qwen2.5-VL-7B-Instruct**：阿里云通义千问的视觉语言模型
- 参数量：约 7B
- 支持图像和文本的多模态理解

### RL 框架
- **veRL**：字节跳动开发的分布式 RL 训练框架
- 支持 FSDP（Fully Sharded Data Parallel）
- 支持 Megatron 后端
- 混合引擎（Hybrid Engine）：同时支持训练和推理

### 推理引擎
- **vLLM**：高性能的 LLM 推理引擎
- 支持张量并行（Tensor Parallelism）
- 支持多轮对话的 rollout

## 项目结构

```
multimodal-search-r1/
├── README.md                    # 项目说明文档
├── mmsearch_r1/                 # 主要代码目录
│   ├── data/                    # 数据集示例
│   ├── monkey_patch/            # 代码补丁
│   ├── prompts/                 # 提示模板
│   ├── scripts/                 # 训练脚本
│   ├── trainer/                 # 训练器实现
│   ├── utils/                   # 工具函数
│   └── workers/                 # 分布式工作器
└── verl/                        # veRL 框架（子模块）
```

## 训练流程

### 1. 数据准备
- 使用 parquet 格式存储数据
- 包含图像和文本的多模态数据
- 支持自定义提示模板

### 2. 模型训练
- 使用 GRPO 算法进行策略优化
- 支持多轮 rollout（最多 3 轮）
- 使用 vLLM 进行高效推理

### 3. 奖励计算
- 格式奖励：检查响应格式
- 正确性奖励：精确匹配或子串匹配
- 搜索惩罚：控制搜索工具的使用

## 关键配置参数

### 训练配置
- `train_batch_size`: 32（训练批量大小）
- `max_prompt_length`: 4096（最大提示长度）
- `max_response_length`: 2048（最大响应长度）
- `response_length_total`: 8192（总响应长度）
- `max_gen_round`: 3（最大生成轮数）

### 模型配置
- `model.path`: Qwen/Qwen2.5-VL-7B-Instruct
- `enable_gradient_checkpointing`: True（梯度检查点）
- `use_remove_padding`: True（移除填充）

### 推理配置
- `rollout.name`: vllm_multiturn_mmsearch
- `tensor_model_parallel_size`: 1（张量并行大小）
- `gpu_memory_utilization`: 0.6（GPU 内存利用率）

## 参考资源

- **论文**：[MMSearch-R1: Incentivizing LMMs to Search](https://arxiv.org/abs/2506.20670)
- **博客**：[lmms-lab.com/posts/mmsearch_r1](https://www.lmms-lab.com/posts/mmsearch_r1)
- **模型**：[huggingface.co/lmms-lab/MMSearch-R1-7B](https://huggingface.co/lmms-lab/MMSearch-R1-7B)
- **数据集**：[huggingface.co/datasets/lmms-lab/FVQA](https://huggingface.co/datasets/lmms-lab/FVQA)
