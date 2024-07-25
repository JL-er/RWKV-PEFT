
base_model='/home/rwkv/JL/model/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
lora_checkpoint='/home/rwkv/JL/out_model/meta_lora_mask/rwkv-0.pth'
output='/home/rwkv/JL/model/meta-lora-mask1.pth'
QUANT='nf4' #follow train
TYPE='lora'
Lora_alpha=128

python merge/merge.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
--type $TYPE \
--lora_alpha $Lora_alpha