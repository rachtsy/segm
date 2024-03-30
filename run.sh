export DATASET=/root/data/

# ##### RPC
python -m segm.train --log-dir /root/checkpoints/segm/6it/ --dataset ade20k \
--backbone deit_tiny_patch16_224 --decoder mask_transformer --model_name rpc --attn_type 'rpc' \
--resume --eval 

##### BASELINE
# python -m segm.train --log-dir /root/checkpoints/segm/baseline/ --dataset ade20k \
# --backbone deit_tiny_patch16_224 --decoder mask_transformer --model_name baseline --attn_type 'softmax' \
# --resume --eval

###### NEUTRENO
# python -m segm.train --log-dir seg_tiny_mask/neutreno --dataset ade20k \
# --backbone deit_tiny_patch16_224 --decoder mask_transformer --model_name neutreno --attn_type 'neutreno-former' --alpha 0.6

