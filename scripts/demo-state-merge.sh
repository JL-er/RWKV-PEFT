base_model='/home/rwkv/JL/model/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth'
state_checkpoint='/home/rwkv/JL/out_model/roleplay/rwkv-0.pth'
output='/home/rwkv/JL/model/roleplay-0.pth'


python merge/merge_state.py --base_model $base_model \
--state_checkpoint $state_checkpoint \
--output $output