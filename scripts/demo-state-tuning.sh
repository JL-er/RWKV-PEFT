load_model='/home/lzy/workspace/rwkv-kit/weight/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
proj_dir='/home/lzy/workspace/RWKV-PEFT/test/output'
data_file='/home/lzy/workspace/RWKV-PEFT/test/input/bad_text_document'


n_layer=24
n_embd=2048

micro_bsz=2
epoch_save=1
epoch_steps=800
ctx_len=512


python train.py --load_model $load_model \
--proj_dir $proj_dir --data_file $data_file \
--data_type binidx --vocab_size 65536 \
--ctx_len $ctx_len --epoch_steps $epoch_steps --epoch_count 10 --epoch_begin 0 --epoch_save $epoch_save --micro_bsz $micro_bsz \
--n_layer $n_layer --n_embd $n_embd \
--pre_ffn 0 --head_qk 0 --lr_init 1 --lr_final 1e-2 --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy single-device --grad_cp 1 \
--rwkv_version "x060" \
--train_type "state"  --dataload pad --fla \
