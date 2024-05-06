base_model='/home/rwkv/JL/model/RWKV-x060-World-1B6-v2-20240208-ctx4096.pth'
state_checkpoint='/home/rwkv/JL/out_model/state/rwkv-4.pth'
output='/home/rwkv/JL/model/state-world-4.pth'


python merge_state.py --base_model $base_model \
--state_checkpoint $state_checkpoint \
--output $output