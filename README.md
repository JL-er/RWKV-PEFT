# RWKV-V4-lora
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
# v4训练配置 RWKV-4-Doctor-7B-lora.pth
```
数据：334MB 问诊对话
数据格式：Patient: {content}\n\nDoctor:{content}\n\n
单卡  
GPU：4090 24G
CPU：64G  
base model：rwkv-v4-7B  
lora：
  --lora_r 64 --lora_alpha 128
```

# 推理配置 RWKV-4-Doctor-7B-lora.pth
```
interface = ":"
user = "Patient"
bot = "Doctor"
GEN_TEMP = 1.0 # It could be a good idea to increase temp when top_p is low
GEN_TOP_P = 0.2 # Reduce top_p (to 0.5, 0.2, 0.1 etc.) for better Q&A accuracy (and less diversity)
GEN_alpha_presence = 0.0 # Presence Penalty
GEN_alpha_frequency = 0.0 # Frequency Penalty
GEN_penalty_decay = 0.996
```
![Uploading image.png…]()



# RWKV-v5-lora
训练技巧：
  标准的全量微调方法： 将数据复制多遍（如果你想炼多个epoch），注意，其中的条目，必须每次用不同的随机排列！ 然后用我这里的 --my_pile_stage 3 --my_exit_tokens xxx --magic_prime xxx 技术，这样采样才是完美无重复的。 学习速率建议 --lr_init 1e-5 --lr_final 1e-5  
  my_exit_tokens = datalen，数据的精确 token 数，在载入数据时会显示 # magic_prime = the largest 3n+2 prime smaller than datalen/ctxlen-1 (= 1498226207/512-1 = 2926222.06 in this case) # use  
  lora：
    --lora_r 64 --lora_alpha 128  r和a 同时增大，越大效果越好但训练速度也会变慢，目前较好参数为64/128
```
python train.py --load_model /home/asd/model/RWKV-5-World-1B5-v2-20231025-ctx4096.pth --proj_dir /home/asd/model --data_file ttt_text_document --data_type binidx --vocab_size 65536 --ctx_len 10 --epoch_steps 10 --epoch_count 100 --epoch_begin 0 --epoch_save 5 --micro_bsz 1 --n_layer 24 --n_embd 2048 --pre_ffn 0 --head_qk 0 --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 1 --lora_load rwkv-0 --lora --lora_r 64 --lora_alpha 128 --lora_dropout 0.01 --lora_parts=att,ffn,time,ln
```

# Merge lora
```
python merge_lora.py --use-gpu 128 /home/asd/model/RWKV-5-World-1B5-v2-20231025-ctx4096.pth img595k/rwkv-0.pth /home/asd/model/RWKV-5-World-1.5B--lora.pth
```
