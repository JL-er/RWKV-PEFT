

QUANT='fp4'


python train.py --load_model /home/rwkv/JL/model/rwkv-x060-7b-world-v2.1-36%trained-20240413-ctx4k.pth \
--proj_dir /home/rwkv/JL/out_model/fp4 --data_file /home/rwkv/JL/data/roleplay \
--data_type binidx --vocab_size 65536 \
--ctx_len 1024 --epoch_steps 10 --epoch_count 1 --epoch_begin 0 --epoch_save 1 --micro_bsz 1 \
--n_layer 32 --n_embd 4096 \
--pre_ffn 0 --head_qk 0 --lr_init 1e-4 --lr_final 1e-4 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
--my_testing "x060" \
--lora_load rwkv-0 --lora --lora_r 64 --lora_alpha 128 --lora_dropout 0.01 --lora_parts=att,ffn,time,ln \
--quant $QUANT