base_model='/home/rwkv/model/rwkv-x070-2b9-world-v3-preview-20250210-ctx4k.pth'
peft_checkpoint='/home/rwkv/JL/out_model/v7-3b-disha/rwkv-0.pth'
output='/home/rwkv/JL/model/v7-3b-disha.pth'


python merge/merge_disha.py --base_model $base_model \
--peft_checkpoint $peft_checkpoint \
--output $output
