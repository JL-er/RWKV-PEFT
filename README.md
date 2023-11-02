# RWKV-v5-lora
```
python train.py --load_model /home/asd/model/RWKV-5-World-1B5-v2-20231025-ctx4096.pth --proj_dir /home/asd/model --data_file ttt_text_document --data_type binidx --vocab_size 65536 --ctx_len 10 --epoch_steps 10 --epoch_count 100 --epoch_begin 0 --epoch_save 5 --micro_bsz 1 --n_layer 24 --n_embd 2048 --pre_ffn 0 --head_qk 0 --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 1 --lora_load rwkv-0 --lora --lora_r 64 --lora_alpha 128 --lora_dropout 0.01 --lora_parts=att,ffn,time,ln
```
