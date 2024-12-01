
<h1 align="center">
  <p><img src="assert/logo.jpg" alt="RWKV-PEFT" width="60px"  style="vertical-align: middle; margin-right: 10px;"/>RWKV-PEFT</p>
</h1>

\[ English | [中文](README_zh.md) \]

RWKV-PEFT is the official implementation for efficient parameter fine-tuning of RWKV5/6 models, supporting various advanced fine-tuning methods across multiple hardware platforms.

# Recent updates
## SFT Train
Relevant parameters, detailed usage reference: scripts/run_sft.sh  
- data_file 'meta-math/MetaMathQA' #You can directly choose the Hugging Face path, or you can choose your own JSON path.  
- data_type sft #Select data type  
- sft_field query answer #Perform retrieval based on the question-and-answer format in the JSON.  
- sft_split "train" #Set the number of data to load: "train" loads all the data, while "train[:1000]" loads only the first 1000 samples.  
```
--data_type sft --sft_field query answer --sft_split "train"
```
## Specific settings for SFT
### RWKV-PEFT/src/rwkv_datasets/SFTdataset.py
```
tokenizer_path = 'RWKV/rwkv-5-world-3b' #Choose a tokenizer (select the official tokenizer)
IGNORE_INDEX = -100 #Padding (do not modify)
EOT_TOKEN = "<|EOT|>" #Set the stop token(s) you need

# Modify the corresponding prompt according to your requirements
PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )
```
> [!TIP]
> Downloading Hugging Face data may time out in China, so you need to add: HF_ENDPOINT="https://hf-mirror.com" sh scripts/run_sft.sh

## [Bone: Block-Affine Adaptation of Large Language Models](https://arxiv.org/pdf/2409.15371)
The paper has been updated. Bone is now a simple and efficient basic PEFT method that is faster and uses less VRAM than LoRA, converges faster, and performs better than PiSSA. The previous version of Bone has been changed to the Bat method.    
scripts:  
```bone_config='{"bone_load":"","bone_r":64}'```update``` bone_config='{"bone_mode":"bone","bone_load":"","bone_r":64}' ``` or``` bone_config='{"bone_mode":"bat","bone_load":"","bone_r":64}' ```


# Installation

> [!IMPORTANT]
> Installation is mandatory.

```bash
git clone https://github.com/JL-er/RWKV-PEFT.git
cd RWKV-PEFT
pip install -r requirements.txt
```

## Web Run
> [!TIP]
> If you are using a cloud server (such as [Vast](https://vast.ai/) or [AutoDL](https://www.autodl.com/)), you can start the Streamlit service by referring to the help documentation on the cloud server's official website.

```bash
gradio web/app.py
```

## Table of Contents
- [Hardware Requirements](#hardware-requirements)
- [Quick Start](#quick-start)
- [Main Features](#main-features)
- [Detailed Configuration](#detailed-configuration)
- [GPU Support](#gpu-support)
- [Citation](#citation)

## Hardware Requirements

The following shows memory usage when using an RTX 4090 (24GB VRAM) + 64GB RAM (with parameters: `--strategy deepspeed_stage_1 --ctx_len 1024 --micro_bsz 1 --lora_r 64`):

|   Model Size   | Full Finetuning | LoRA/PISSA | QLoRA/QPISSA | State Tuning |
|---------------|-----------------|------------|--------------|--------------|
| RWKV6-1.6B    | OOM            | 7.4GB      | 5.6GB        | 6.4GB        |
| RWKV6-3B      | OOM            | 12.1GB     | 8.2GB        | 9.4GB        |
| RWKV6-7B      | OOM            | 23.7GB*    | 14.9GB**     | 18.1GB       |

Note:
* OOM when batch size is 8
** Requires 19.5GB VRAM when batch size is 8

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run example script:
```bash
sh scripts/run_lora.sh
```
Note: Please refer to the RWKV official tutorial for detailed data preparation

3. Start with web GUI:
> [!TIP]
> If you're using cloud services (such as [Vast](https://vast.ai/) or [AutoDL](https://www.autodl.com/)), you'll need to enable web port access according to your service provider's instructions.

```bash
streamlit run web/app.py
```

## Main Features

- **Multiple Fine-tuning Methods**: Supports LoRA, PISSA, Bone, State Tuning, etc.
- **Quantized Training**: Supports INT8/NF4 quantization for significant VRAM reduction
- **Flexible Data Loading**: Supports various data sampling strategies 
- **Memory Optimization**: Multiple DeepSpeed strategies available
- **Loss Masking**: Supports loss masking for QA dialogue and padding
- **Infinite Context Training**: Supports infctx training mode, utilizing RWKV's constant memory usage advantage to train with "infinite" context under limited resources
- **Multi-Hardware Support**: RWKV-PEFT officially supports NVIDIA, AMD, Moore Threads, Musa, Iluvatar CoreX, and other hardware platforms. Ascend NPU implementation will be available later. Note: Currently we only support issues for NVIDIA hardware
- **RWKV-FLA Efficient Training**: rwkv-fla is a Triton-based linear attention operator that can run efficiently on hardware without CUDA support

## Detailed Configuration

### 1. PEFT Method Selection
```bash
--peft bone --bone_config $lora_config
```

### 2. Training Parts Selection
```bash
--train_parts ["time", "ln"]
```
- Available parts: emb, head, time, ln
- Default training: time, ln (small parameter ratio)

### 3. Quantized Training
```bash
--quant int8/nf4
```

### 4. Infinite Length Training (infctx)
```bash
--train_type infctx --chunk_ctx 512 --ctx_len 2048
```
- ctx_len: Target training length
- chunk_ctx: Slice length, must be smaller than ctx_len

### 5. Data Loading Strategy
```bash
--dataload pad
```
- get: Default random sampling (RWKV-LM style)
- pad: Fixed-length padding sampling
- only: Single data sampling (only supports bsz=1)

### 6. DeepSpeed Strategy
```bash
--strategy deepspeed_stage_1
```
Available strategies:
- deepspeed_stage_1: Preferred option
- deepspeed_stage_2/3: For large models or full fine-tuning
- deepspeed_stage_2_offload
- deepspeed_stage_3_offload

### 7. FLA Operator
By default, RWKV-PEFT uses custom CUDA kernels for wkv computation.
However, you can use `--fla` to enable the Triton kernel:
```
--fla
```

## GPU Support

- NVIDIA: CUDA
- Intel, Moore Threads, Musa, Iluvatar CoreX: FLA, which means you need to pass `--fla`
- Ascend: CANN (soon)

## Citation

If you find this project helpful, please cite our work:
```bib
@misc{kang2024boneblockaffineadaptationlarge,
      title={Bone: Block-Affine Adaptation of Large Language Models}, 
      author={Jiale Kang},
      year={2024},
      eprint={2409.15371},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.15371}, 
}