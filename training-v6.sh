
# ------------------通用参数----------------------
# 微调方式
FINETUNE_MODE="lora"
# 模型路径
MODEL_PATH=model/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth
# 数据路径
DATA_PATH=data/xuexue
# 输出路径
OUTPUT_PATH=output
# 训练的回合数
EPOCH_COUNT=100
# 回合步数
EPOCH_STEPS=200
# 上下文长度
CTX_LEN=4096
# 精度
PRECISION=bf16
# 初始学习率
LR_INIT=3e-4
# 最终学习率
LR_FINAL=3e-4
# 显卡数量
GPU_COUNT=1
# 微批次大小
MICRO_BSZ=1
# 模型保存间隔
EPOCH_SAVE=1
# 前缀网络预处理
PRE_FFN=0
# 梯度累计
MINI_BSZ=1
# 优化策略
STRATEGY=deepspeed_stage_2
# 梯度复制
GRAD_CP=1

# ------------------不常用训练参数----------------------
# 开始训练的回合，可以用来恢复训练
EPOCH_BEGIN=0
# 词表大小
VOCAB_SIZE=65536
# 嵌入维度
EMBD_SIZE=2048
# 嵌入层
N_LAYER=24
# Head QK
HEAD_QK=0
# Bata1
BETA1=0.9
# Bata2
BETA2=0.99
# 预热步数
WARMUP_STEPS=0
# ADAM epsilon
ADAM_EPS=1e-8

# ------------------Lora和Pissa设置参数----------------------
# lora_parts
lora_parts=att,ffn,time,ln
# LORA模型路径，代表从哪个LORA模型开始微调
lora_load="rwkv-0"
# LORA模型的r值
lora_r=64
# LORA模型的alpha值
lora_alpha=128
# LORA模型的dropout值
lora_dropout=0.01
# pissa的svd迭代次数
svd_niter=4

# ------------------lisa设置参数----------------------
# LISA模型的r值
lisa_r=2
# LISA模型的k值
lisa_k=100


if [ "$FINETUNE_MODE" == "lora" ]; then
   python3 train.py --load_model $MODEL_PATH \
    --proj_dir $OUTPUT_PATH --data_file $DATA_PATH \
    --data_type binidx --vocab_size $VOCAB_SIZE \
    --ctx_len $CTX_LEN --epoch_steps $EPOCH_STEPS --epoch_count $EPOCH_COUNT --epoch_begin $EPOCH_BEGIN --epoch_save $EPOCH_SAVE --micro_bsz $MICRO_BSZ \
    --n_layer $N_LAYER --n_embd $EMBD_SIZE \
    --pre_ffn $PRE_FFN --head_qk $HEAD_QK --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps $WARMUP_STEPS --beta1 $BETA1 --beta2 $BETA2 --adam_eps 1e-8 \
    --accelerator gpu --devices $GPU_COUNT --precision $PRECISION --strategy $STRATEGY --grad_cp $GRAD_CP \
    --accumulate_grad_batches $MINI_BSZ \
    --my_testing "x060" \
    --lora_load $lora_load --lora --lora_r $lora_r --lora_alpha $lora_alpha --lora_dropout $lora_dropout --lora_parts=$lora_parts
else if [ "$FINETUNE_MODE" == "lisa" ]; then
   python3 train.py --load_model $MODEL_PATH \
    --proj_dir $OUTPUT_PATH --data_file $DATA_PATH \
    --data_type binidx --vocab_size $VOCAB_SIZE \
    --ctx_len $CTX_LEN --epoch_steps $EPOCH_STEPS --epoch_count $EPOCH_COUNT --epoch_begin $EPOCH_BEGIN --epoch_save $EPOCH_SAVE --micro_bsz $MICRO_BSZ \
    --n_layer $N_LAYER --n_embd $EMBD_SIZE \
    --pre_ffn $PRE_FFN --head_qk $HEAD_QK --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps $WARMUP_STEPS --beta1 $BETA1 --beta2 $BETA2 --adam_eps $ADAM_EPS \
    --accelerator gpu --devices $GPU_COUNT --precision $PRECISION --strategy $STRATEGY --grad_cp $GRAD_CP \
    --accumulate_grad_batches $MINI_BSZ \
    --my_testing "x060" \
    --LISA --lisa_r $lisa_r --lisa_k $lisa_k
else if [ "$FINETUNE_MODE" == "pissa" ]; then
   python3 train.py --load_model $MODEL_PATH \
    --proj_dir $OUTPUT_PATH --data_file $DATA_PATH \
    --data_type binidx --vocab_size $VOCAB_SIZE \
    --ctx_len $CTX_LEN --epoch_steps $EPOCH_STEPS --epoch_count $EPOCH_COUNT --epoch_begin $EPOCH_BEGIN --epoch_save $EPOCH_SAVE --micro_bsz $MICRO_BSZ \
    --n_layer $N_LAYER --n_embd $EMBD_SIZE \
    --pre_ffn $PRE_FFN --head_qk $HEAD_QK --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps $WARMUP_STEPS --beta1 $BETA1 --beta2 $BETA2 --adam_eps $ADAM_EPS \
    --accelerator gpu --devices $GPU_COUNT --precision $PRECISION --strategy $STRATEGY --grad_cp $GRAD_CP \
    --accumulate_grad_batches $MINI_BSZ \
    --my_testing "x060" \
    --lora_load $lora_load --lora --lora_r $lora_r --lora_alpha $lora_alpha --lora_dropout $lora_dropout --lora_parts=$lora_parts \
    --PISSA --svd_niter $svd_niter
fi
fi
fi