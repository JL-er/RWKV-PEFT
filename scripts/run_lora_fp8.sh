load_model='/home/client/Projects/RWKV-LM-RLHF/v6/models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth'
proj_dir='/home/client/Projects/RWKV/test'
data_file='/home/client/Projects/RWKV/RWKV5-LM-LoRA/RWKV-v6/dataset/test'

n_layer=32
n_embd=4096

micro_bsz=4
epoch_save=1
epoch_steps=200 #6171
ctx_len=1024

lora_config='{"lora_load":"","lora_r":32,"lora_alpha":32,"lora_dropout":0.0,"lora_parts":"att,ffn"}'


python train.py --load_model $load_model \
--proj_dir $proj_dir --data_file $data_file \
--data_type binidx --vocab_size 65536 \
--ctx_len $ctx_len --epoch_steps $epoch_steps --epoch_count 1 --epoch_begin 0 --epoch_save $epoch_save --micro_bsz $micro_bsz \
--n_layer $n_layer --n_embd $n_embd \
--pre_ffn 0 --head_qk 0 --lr_init 2e-5 --lr_final 2e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
--my_testing "x060" \
--peft lora --lora_config $lora_config --train_parts [] \
--wandb "RWKV-PEFT FP8 Test LoRA" \
--quant "fp8" 
