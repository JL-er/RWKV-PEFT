

base_model='/home/rwkv/model/rwkv-x070-2b9-world-v3-preview-20250210-ctx4k.pth'
pissa_init='/home/rwkv/JL/out_model/v7-3b-pissa-math/init_pissa.pth'
pissa_checkpoint='/home/rwkv/JL/out_model/v7-3b-pissa-math/rwkv-0.pth'
output='/home/rwkv/JL/model/v7-math-pissa-3b.pth'
TYPE='pissa'

python merge/merge.py --base_model $base_model \
--lora_init $pissa_init \
--lora_checkpoint $pissa_checkpoint \
--output $output \
--type $TYPE 