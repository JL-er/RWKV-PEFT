<h1 align="center">
  <p><img src="assert/logo.jpg" alt="RWKV-PEFT" width="60px"  style="vertical-align: middle; margin-right: 10px;"/>RWKV-PEFT</p>
</h1>

\[ [English](README.md) | [中文] \]

RWKV-PEFT 是一个旨在为 RWKV5/6 模型实现高效参数微调的官方实现，支持在多种硬件上实现多种先进的微调方法。


# Installation

> [!IMPORTANT]
> 不可跳过

```bash
git clone https://github.com/JL-er/RWKV-PEFT.git
cd RWKV-PEFT
pip install -r requirements.txt
```

## Web Run
> [!TIP]
> 如果你想使用云服务器运行streamlit (如 [Vast](https://vast.ai/) or [AutoDL](https://www.autodl.com/)), 你需要查看云服务器平台教程进行配置

```bash
gradio web/app.py
```

## 目录
- [硬件需求](#硬件需求)
- [快速开始](#快速开始)
- [主要特性](#主要特性)
- [详细配置说明](#详细配置说明)
- [GPU支持情况](#gpu支持情况)
- [引用](#引用)

## 硬件需求

以下是使用 RTX 4090 (24GB显存) + 64GB内存时的显存占用情况（参数配置：`--strategy deepspeed_stage_1 --ctx_len 1024 --micro_bsz 1 --lora_r 64`）：

|   模型规模   | 全量微调 | LoRA/PISSA | QLoRA/QPISSA | State Tuning |
|-------------|----------|------------|--------------|--------------|
| RWKV6-1.6B  | 显存溢出   | 7.4GB      | 5.6GB        | 6.4GB        |
| RWKV6-3B    | 显存溢出   | 12.1GB     | 8.2GB        | 9.4GB        |
| RWKV6-7B    | 显存溢出   | 23.7GB*    | 14.9GB**     | 18.1GB       |

注：
* 批次大小为8时会显存溢出
* 批次大小为8时需要19.5GB显存

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行示例脚本：
```bash
sh scripts/run_lora.sh
```
注：具体数据准备方法请参考RWKV官方教程

3. 使用 web gui 开始：
> [!TIP]
> 如果您使用云服务 (such as [Vast](https://vast.ai/) or [AutoDL](https://www.autodl.com/)), 您需要参考相关服务商的提示，开启网页端口业务。

```bash
streamlit run web/app.py
```

## 主要特性

- **多种微调方法**：支持LoRA、PISSA、Bone, State Tuning等
- **量化训练**：支持INT8/NF4量化，显著降低显存占用
- **灵活的数据加载**：支持多种数据采样策略
- **显存优化**：多种DeepSpeed策略可选
- **损失Mask**：支持QA对话和填充部分的损失Mask
- **无限长度训练**：支持infctx训练模式, 此模式利用了RWKV恒定显存占用的优势，在有限的资源下训练“无限”上下文
- **支持多种硬件**：目前，RWKV-PEFT 官方支持 NVIDIA, AMD, 摩尔线程，沐曦，天数智芯等多种硬件平台, 昇腾NPU的实现会在后期实现。注意：目前我们只支持 NVIDIA 的 issue 请求。
- **使用rwkv-fla高效训练**: rwkv-fla是基于triton的线性注意力算子，可以在不支持cuda的硬件上高效率运行。

## 详细配置说明

### 1. PEFT方法选择
```bash
--peft bone --bone_config $lora_config
```

### 2. 训练部分选择
```bash
--train_parts ["time", "ln"]
```
- 可选部分：emb、head、time、ln
- 默认训练：time、ln（参数量占比小）

### 3. 量化训练
```bash
--quant int8/nf4
```

### 4. 无限长度训练（infctx）
```bash
--train_type infctx --chunk_ctx 512 --ctx_len 2048
```
- ctx_len：目标训练长度
- chunk_ctx：切片长度，需小于ctx_len

### 5. 数据加载策略
```bash
--dataload pad
```
- get：默认随机采样（RWKV-LM方式）
- pad：固定长度填充采样
- only：单条数据采样（仅支持bsz=1）

### 6. DeepSpeed策略
```bash
--strategy deepspeed_stage_1
```
可选策略：
- deepspeed_stage_1：优先使用
- deepspeed_stage_2/3：大模型或全量微调时使用
- deepspeed_stage_2_offload
- deepspeed_stage_3_offload

### 7. FLA算子
默认情况下， RWKV-PEFT 会使用自定义的cuda内核来实现wkv计算。 但您也可以使用`--fla`来开启Triton内核。
```
--fla
```
## GPU支持情况

- NVIDIA: CUDA
- Intel、摩尔线程、沐曦、天数智芯: FLA, 这意味着你需要手动传入 `--fla`
- 昇腾: CANN(soon)

## 引用

如果您觉得本项目对您有帮助，请引用我们的工作：

```bibtex
@misc{kang2024boneblockaffinetransformation,
      title={Bone: Block Affine Transformation as Parameter Efficient Fine-tuning Methods for Large Language Models},
      author={Jiale Kang},
      year={2024},
      eprint={2409.15371},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.15371}
}
```