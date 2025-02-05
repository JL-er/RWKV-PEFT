base_model='/home/rwkv/JL/model/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
lora_checkpoint='/home/rwkv/JL/out_model/metalora-1.6b/rwkv-0.pth'
output='/home/rwkv/JL/model/metalora-1.6b.pth'
TYPE='lora'
Lora_alpha=32

python merge/merge.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
--type $TYPE \
--lora_alpha $Lora_alpha