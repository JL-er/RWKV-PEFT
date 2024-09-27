

base_model='/home/rwkv/JL/model/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
lora_init='/home/rwkv/JL/out_model/metapissa-1.6b/init_pissa.pth'
lora_checkpoint='/home/rwkv/JL/out_model/metapissa-1.6b/rwkv-0.pth'
output='/home/rwkv/JL/model/metapissa-1.6b.pth'
TYPE='pissa'

python merge/merge.py --base_model $base_model \
--lora_init $lora_init \
--lora_checkpoint $lora_checkpoint \
--output $output \
--type $TYPE 