# load_model='/home/lzy/workspace/rwkv-kit/weight/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
proj_dir='/home/lzy/workspace/RWKV-PEFT/test/output'
data_file='/home/lzy/workspace/RWKV-PEFT/test/input/bad_text_document'


n_layer=12
n_embd=768

micro_bsz=2
epoch_save=10
epoch_steps=800
ctx_len=512
LR_INIT="6e-4"
LR_FINAL="6e-5"

# QUANT='nf4' 

python train.py \
--proj_dir $proj_dir --data_file $data_file \
--data_type binidx --vocab_size 65536 \
--ctx_len $ctx_len --epoch_steps $epoch_steps --epoch_count 1 --epoch_begin 0 --epoch_save $epoch_save --micro_bsz $micro_bsz \
--n_layer $n_layer --n_embd $n_embd \
--pre_ffn 0 --head_qk 0 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator xpu --devices 1 --precision bf16 --strategy single --grad_cp 1 \
--my_testing "x060" \
--train_type "finetune"  --dataload pad --fla --wandb peft-test-platform