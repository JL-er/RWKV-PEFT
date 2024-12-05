load_model='/home/rwkv/JL/model/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth'
proj_dir='/home/rwkv/JL/out_model/bone'
data_file='meta-math/MetaMathQA'

n_layer=32
n_embd=2560

micro_bsz=4
epoch_save=1
epoch_steps=200 #6171
ctx_len=512

bone_config='{"bone_mode":"bone","bone_load":"","bone_r":64}'


python train.py --load_model $load_model \
--proj_dir $proj_dir --data_file $data_file \
--vocab_size 65536 \
--n_layer $n_layer --n_embd $n_embd \
--ctx_len $ctx_len --micro_bsz $micro_bsz \
--epoch_steps $epoch_steps --epoch_count 1 --epoch_begin 0 --epoch_save $epoch_save \
--lr_init 1e-4 --lr_final 1e-4 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
--my_testing "x060" \
--peft bone --bone_config $bone_config --data_shuffle 0 \
--data_type sft --sft_field query response --sft_split "train[:1000]"