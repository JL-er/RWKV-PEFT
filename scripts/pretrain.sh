load_model=''
proj_dir='/home/rwkv/JL/out_model/pretrain'
data_file='/home/rwkv/JL/data/MetaMathQA'

n_layer=12
n_embd=768

micro_bsz=4
epoch_save=1
epoch_steps=1 #6171
ctx_len=512


python train.py \
--proj_dir $proj_dir --data_file $data_file \
--vocab_size 65536 \
--n_layer $n_layer --n_embd $n_embd \
--ctx_len $ctx_len --micro_bsz $micro_bsz \
--epoch_steps $epoch_steps --epoch_count 3 --epoch_begin 0 --epoch_save $epoch_save \
--lr_init 3e-4 --lr_final 1e-4 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
--my_testing "x060" \
--data_type sft --sft_field query response --sft_split "train[:1000]"