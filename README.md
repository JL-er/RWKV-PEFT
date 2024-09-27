
<h1 align="center"> <p>🦚 RWKV-PEFT</p></h1>

\[ English | [中文](README_zh.md) \]

# Release
- infctx
- fla --fla
- State tuning
- Quant(QPissa,QLora) --quant int8/nf4
- Pissa
- Lisa
- Lora
- dataload(get、pad、only)
### High performance on consumer hardware

Consider the memory requirements for training the following models with an 4090 24GB GPU with 64GB of CPU RAM.(--strategy deepspeed_stage_1 --ctx_len 1024 --micro_bsz 1 --lora_r 64)

|   Model         | Full Finetuning | lora/pissa  | Qlora/Qpissa | State tuning |
| --------- | ---- | ---- | ---- | ---- |
| RWKV6-1.6B | OOM GPU | 7.4GB GPU | 5.6GB GPU | 6.4GB GPU |
| RWKV6-3B | OOM GPU | 12.1GB GPU | 8.2GB GPU | 9.4GB GPU |
| RWKV6-7B | OOM GPU | 23.7GB GPU(bsz 8 OOM) | 14.9GB GPU(bsz 8 need 19.5GB) | 18.1GB GPU |
#### Quant State Tuning
- strategy deepspeed_stage_1
- ctx_len 1024
- micro_bsz 1
- 4090 24G

|   Model         | bf16 | int8  | nf4/fp4/4bit |
| --------- | ---- | ---- | ---- |
| RWKV6-1.6B | 6.1GB GPU | 4.7GB GPU | 4.1GB GPU |
| RWKV6-3B | 9.1GB GPU | 6.5GB GPU | 5.2GB GPU |
| RWKV6-7B | 17.8GB GPU | 11.9GB GPU | 8.5GB GPU |
| RWKV6-14B | xxGB GPU | xxGB GPU | xxGB GPU |
# Usage
sh demo/demo-xxxx.sh
### --train_type
"--quant (infctx state)"
### infctx train
"--train_type infctx --chunk_ctx 512" 
"chunk_ctx" represents the chunk length, while "ctx_len" stands for the total length of the data.
Due to the lack of gradients in the wkv6state operator, I now recommend using fla instead.
```
python train.py --load_model /home/rwkv/JL/model/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth \
--proj_dir /home/rwkv/JL/out_model/state --data_file /home/rwkv/JL/data/roleplay \
--data_type binidx --vocab_size 65536 \
--ctx_len 2048 --epoch_steps 1000 --epoch_count 100 --epoch_begin 0 --epoch_save 1 --micro_bsz 4 \
--n_layer 24 --n_embd 2048 \
--pre_ffn 0 --head_qk 0 --lr_init 1 --lr_final 1e-1 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
--my_testing "x060" \
--train_type infctx --chunk_ctx 512 --fla
```
### fla
pip install triton==2.2.0
add "--fla" to utilize."FLA" doesn't need to be compiled, make sure Triton is installed before using it.
https://github.com/sustcsonglin/flash-linear-attention.git
### State Tuning
add "--train_type state " to utilize quantization State Tuning.  
This project's state tuning currently only supports training the state. You can refer to the state tuning in the demo for configuration. When saving weights, only the state is retained, so you need to use the state merge from the demo for merging. The advantage is that the saved weight files are very small. Any user who uses the same base model as you trained can merge and experience the same training results.
```
python train.py --load_model /home/rwkv/JL/model/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth \
--proj_dir /home/rwkv/JL/out_model/state --data_file /home/rwkv/JL/data/roleplay \
--data_type binidx --vocab_size 65536 \
--ctx_len 2048 --epoch_steps 1000 --epoch_count 100 --epoch_begin 0 --epoch_save 1 --micro_bsz 4 \
--n_layer 24 --n_embd 2048 \
--pre_ffn 0 --head_qk 0 --lr_init 1 --lr_final 1e-1 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
--my_testing "x060" \
--train_type state
```

### Quant Train
You just need to add "--quant (int8 4bit nf4 fp4)" to utilize quantization fine-tuning.
You can also use "sh demo-pissa.sh" for a quick start.Then use "sh demo-pissa-merge.sh" for merging.

### PISSA
PISSA is better than LISA  
--lora_alpha 128 --lora_dropout 0.01 (These two parameters do not work.)

```
python train.py --load_model /home/rwkv/JL/model/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth \
--proj_dir /home/rwkv/JL/out_model/lisa-l2 --data_file /home/rwkv/JL/data/roleplay \
--data_type binidx --vocab_size 65536 \
--ctx_len 2048 --epoch_steps 1000 --epoch_count 100 --epoch_begin 0 --epoch_save 1 --micro_bsz 4 \
--n_layer 24 --n_embd 2048 \
--pre_ffn 0 --head_qk 0 --lr_init 1e-4 --lr_final 1e-4 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
--my_testing "x060" \
--lora_load rwkv-0 --lora --lora_r 64 --lora_alpha 128 --lora_dropout 0.01 --lora_parts=att,ffn,time,ln \
--PISSA --svd_niter 4
```
PISSA merge (you need merge init_lora and rwkv-0)
```
python merge_pissa.py --use-gpu /home/rwkv/JL/model/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth /home/rwkv/JL/out_model/lora-1e-4/init_lora.pth /home/rwkv/JL/out_model/lora-1e-4/rwkv-0.pth  /home/rwkv/JL/model/pissa.pth
```

### LISA
LISA is faster and more memory-efficient than LoRA.  
In the context of the LISA algorithm, lisa_r determines how many layers are updated simultaneously, while lisa_k determines how often the algorithm re-selects layers for updating.

```
python train.py --load_model /home/rwkv/JL/model/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth \
--proj_dir /home/rwkv/JL/out_model/lisa-l2 --data_file /home/rwkv/JL/data/roleplay \
--data_type binidx --vocab_size 65536 \
--ctx_len 2048 --epoch_steps 1000 --epoch_count 100 --epoch_begin 0 --epoch_save 1 --micro_bsz 4 \
--n_layer 24 --n_embd 2048 \
--pre_ffn 0 --head_qk 0 --lr_init 1e-4 --lr_final 1e-4 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
--my_testing "x060" \
--LISA --lisa_r 2 --lisa_k 100
```

### RWKV-v6-lora
只需要再v5指令基础上增加 --my_testing "x060"
```
python train.py --load_model /home/rwkv/JL/model/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth \
--proj_dir /home/rwkv/JL/out_model --data_file /home/rwkv/JL/data/minipile \
--data_type binidx --vocab_size 65536 \
--ctx_len 2048 --epoch_steps 8000 --epoch_count 100 --epoch_begin 0 --epoch_save 5 --micro_bsz 4 \
--n_layer 24 --n_embd 2048 \
--pre_ffn 0 --head_qk 0 --lr_init 3e-4 --lr_final 3e-4 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
--my_testing "x060" \
--wandb rwkv \
--lora_load rwkv-0 --lora --lora_r 64 --lora_alpha 128 --lora_dropout 0.01 --lora_parts=att,ffn,time,ln
```
### Merge lora
```
python merge_lora.py --use-gpu 128 /home/asd/model/RWKV-5-World-1B5-v2-20231025-ctx4096.pth img595k/rwkv-0.pth /home/asd/model/RWKV-5-World-1.5B--lora.pth
```

### RWKV-v5-lora
训练技巧：
  标准的全量微调方法： 将数据复制多遍（如果你想炼多个epoch），注意，其中的条目，必须每次用不同的随机排列！ 然后用我这里的 --my_pile_stage 3 --my_exit_tokens xxx --magic_prime xxx 技术，这样采样才是完美无重复的。 学习速率建议 --lr_init 1e-5 --lr_final 1e-5  
  my_exit_tokens = datalen，数据的精确 token 数，在载入数据时会显示 # magic_prime = the largest 3n+2 prime smaller than datalen/ctxlen-1 (= 1498226207/512-1 = 2926222.06 in this case) # use  
  lora：
    --lora_r 64 --lora_alpha 128  r和a 同时增大，越大效果越好但训练速度也会变慢，目前较好参数为64/128
```
python train.py --load_model /home/asd/model/RWKV-5-World-1B5-v2-20231025-ctx4096.pth \
--proj_dir /home/asd/model --data_file ttt_text_document \
--data_type binidx --vocab_size 65536 \
--ctx_len 10 --epoch_steps 10 --epoch_count 100 --epoch_begin 0 --epoch_save 5 --micro_bsz 1 \
--n_layer 24 --n_embd 2048 \
--pre_ffn 0 --head_qk 0 --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 1 \
--lora_load rwkv-0 --lora --lora_r 64 --lora_alpha 128 --lora_dropout 0.01 --lora_parts=att,ffn,time,ln
```
### RWKV-V4-lora
源代码地址：https://github.com/Blealtan/RWKV-LM-LoRA
```
python3 train.py \
  --load_model <pretrained base model> \
  --proj_dir <place to save checkpoints> \
  --data_file <data for finetune> \
  --data_type <data type for finetune, recommend binidx> \
  --vocab_size 50277 --ctx_len 1024 --epoch_steps 1000 --epoch_count 1000 --epoch_begin 0 --epoch_save 5 --micro_bsz 2 --accumulate_grad_batches 4 \
  --n_layer 24 --n_embd 1024 --pre_ffn 0 --head_qk 0 --lr_init 1e-4 --lr_final 1e-4 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0 \ # all your familiar options
  --lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.01 \
  --lora_load <lora checkpoint to continue training> \ # optional
  --lora_parts=att,ffn,time,ln # configure which parts to finetune
```
