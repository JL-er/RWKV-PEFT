QUANT='fp4'
TYPE='lora'
Lora_alpha=128
python merge.py --base_model /home/rwkv/JL/model/rwkv-x060-7b-world-v2.1-36%trained-20240413-ctx4k.pth \
--lora_checkpoint /home/rwkv/JL/out_model/fp4/rwkv-0.pth \
--output /home/rwkv/JL/model/fp4-world.pth \
--quant $QUANT \
--type $TYPE \
--lora_alpha $Lora_alpha