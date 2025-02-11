load_model='xxx/RWKV7.pth'
proj_dir='xxx/xxx'
data_file='xxx/xxx'

n_layer=32
n_embd=2560


micro_bsz=8
epoch_save=1
epoch_steps=1000
ctx_len=512

pissa_config='{"pissa_load":"","pissa_init":"","pissa_r":32,"svd_niter":4}'


python train.py --load_model $load_model \
--proj_dir $proj_dir --data_file $data_file \
--vocab_size 65536 \
--n_layer $n_layer --n_embd $n_embd \
--data_type binidx --dataload pad --loss_mask pad \
--ctx_len $ctx_len --micro_bsz $micro_bsz \
--epoch_steps $epoch_steps --epoch_count 1 --epoch_begin 0 --epoch_save $epoch_save \
--lr_init 2e-5 --lr_final 2e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
--my_testing "x070" \
--peft pissa --pissa_config $pissa_config