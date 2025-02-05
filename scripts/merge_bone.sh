
base_model='/home/rwkv/JL/model/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth'
bone_checkpoint='/home/rwkv/JL/out_model/peft-bone64/rwkv-0.pth'
output='/home/rwkv/JL/model/peft-bone64.pth'


python merge/merge_bone.py --base_model $base_model \
--lora_checkpoint $bone_checkpoint \
--output $output \
