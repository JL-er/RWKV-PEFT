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
# v4训练配置
```
GPU：4090 24G
CPU：64G  
base model：rwkv-v4-7B  
lora：
  --lora_r 64 --lora_alpha 128
```
效果展示
![Uploading image.png…]()


# RWKV-v5-lora
```
python train.py --load_model /home/asd/model/RWKV-5-World-1B5-v2-20231025-ctx4096.pth --proj_dir /home/asd/model --data_file ttt_text_document --data_type binidx --vocab_size 65536 --ctx_len 10 --epoch_steps 10 --epoch_count 100 --epoch_begin 0 --epoch_save 5 --micro_bsz 1 --n_layer 24 --n_embd 2048 --pre_ffn 0 --head_qk 0 --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 1 --lora_load rwkv-0 --lora --lora_r 64 --lora_alpha 128 --lora_dropout 0.01 --lora_parts=att,ffn,time,ln
```

# Merge lora
```
python merge_lora.py --use-gpu 128 /home/asd/model/RWKV-5-World-1B5-v2-20231025-ctx4096.pth img595k/rwkv-0.pth /home/asd/model/RWKV-5-World-1.5B--lora.pth
```
