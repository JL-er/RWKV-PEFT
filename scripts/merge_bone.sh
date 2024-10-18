
base_model='/home/rwkv/JL/model/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
bone_checkpoint='/home/rwkv/JL/out_model/metabone-1.6b/rwkv-0.pth'
output='/home/rwkv/JL/model/metabone-1.6b.pth'


python merge/merge_bone.py --base_model $base_model \
--lora_checkpoint $bone_checkpoint \
--output $output \
