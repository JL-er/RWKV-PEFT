
QUANT='nf4'

python merge.py --base_model /home/rwkv/JL/model/rwkv-x060-7b-world-v2.1-36%trained-20240413-ctx4k.pth \
--lora_init /home/rwkv/JL/out_model/nf4/init_lora.pth \
--lora_checkpoint /home/rwkv/JL/out_model/nf4/rwkv-0.pth \
--output /home/rwkv/JL/model/nf4-world.pth \
--quant $QUANT \
--type pissa 