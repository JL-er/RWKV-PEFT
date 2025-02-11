<h1 align="center">
  <p><img src="assert/logo.jpg" alt="RWKV-PEFT" width="60px"  style="vertical-align: middle; margin-right: 10px;"/>RWKV-PEFT</p>
</h1>

\[ [English](README.md) | [中文] \]

RWKV-PEFT 是一个旨在为 RWKV 模型实现高效参数微调的官方实现，支持在多种硬件上实现多种先进的微调方法。

# 最近更新
## 支持 v7和一些代码调整
 - 1.移除了 `--fla` 并增加了 `--op cuda/fla/triton`. 现在你可以在--op中自由的选择算子。默认推荐cuda，如果你想要使用state tuning 请设置`--op fla` 和 `--train_type state`.
 - 2.修改名称 Bone to DiSHA:  
``` disha_config='{"mode":"bone","load":"","r":64}' ```  
你仍可有选择两种不同的模式 `bone` or `bat`
- 3.模型代码更干净且容易迁移。 详细查看文件 `rwkvt` .
- 4.移除了简易的可视化训练，后续会有专门的程序支持可视化训练

``` --my_testing "x070" ```
## SFT训练
相关参数,详细使用参考scripts/run_sft.sh  
--data_file 'meta-math/MetaMathQA' 可直接选择huggingface路径，也可选择自己的json路径  
--data_type sft 选择数据类型  
--sft_field query response 根据json中问答格式进行检索  
--sft_split "train" 设置加载数据数量"train"全部加载，"train[:1000]"只加载1000条数据  
```
--data_type sft --sft_field query response --sft_split "train"
```
### SFT具体设置
#### RWKV-PEFT/src/rwkv_datasets/SFTdataset.py
```
tokenizer_path = 'RWKV/rwkv-5-world-3b' #选择分词器（选择官方分词器）
IGNORE_INDEX = -100 #填充（请勿修改）
EOT_TOKEN = "\x17" #设置你需要的停止符

# 根据需求修改对应的prompt
PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )
```
> [!TIP]
> 中国网络下载huggingface数据会超时，所以你需要添加:HF_ENDPOINT="https://hf-mirror.com"  
>```HF_ENDPOINT="https://hf-mirror.com" sh scripts/run_sft.sh```

## Bone: Block-Affine Adaptation of Large Language Models [Paper](https://arxiv.org/pdf/2409.15371)
论文更新，现在DiSHA(bone)是一个简单高效基础PEFT方法，比LoRA更快更省显存，比PiSSA收敛更快表现更好。
scripts:  
DiSHA(Bone):``` disha_config='{"mode":"bone","load":"","r":64}' ``` 
DiSHA(Bat):``` disha_config='{"mode":"bat","load":"","r":64}' ```


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
> Coming Soon!

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
--peft disha --disha_config $disha_config
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
```bib
@misc{kang2025dishadimensionshardingadaptationlarge,
      title={DiSHA: Dimension-Sharding Adaptation of Large Language Models with Fast Convergence and Fast Computation}, 
      author={Jiale Kang},
      year={2025},
      eprint={2409.15371},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.15371}, 
}