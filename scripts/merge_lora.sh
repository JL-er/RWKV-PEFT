base_model='/home/rwkv/model/rwkv-x070-2b9-world-v3-preview-20250210-ctx4k.pth'
lora_checkpoint='/home/rwkv/JL/out_model/v7-3b-lora-math/rwkv-0.pth'
output='/home/rwkv/JL/model/v7-math-lora-3b.pth'
TYPE='lora'
Lora_alpha=72

python merge/merge.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
--type $TYPE \
--lora_alpha $Lora_alpha