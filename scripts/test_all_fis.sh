path='/home/zkl/Documents/Data/GDLT_data/VST/RG'
gpu='0'

CUDA_VISIBLE_DEVICES=${gpu} python main.py \
    --video-path ${path}'/swintx_avg_fps25_clip32' \
    --train-label-path ${path}'/train.txt' \
    --test-label-path ${path}'/test.txt' \
    --model-name PCS \
    --submodel-name cofinal \
    --action-type PCS \
    --lr 1e-2 --epoch 200 \
    --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 \
    --loss_align 1 --activate_regular_restrictions 3 --n_head 7 --hidden_dim 448 \
    --test --ckpt ./temp_log/fis-v/VST/cofinal/TES_best.pkl


CUDA_VISIBLE_DEVICES=${gpu} python main.py \
    --video-path ${path}'/swintx_avg_fps25_clip32' \
    --train-label-path ${path}'/train.txt' \
    --test-label-path ${path}'/test.txt' \
    --model-name TES \
    --submodel-name cofinal \
    --action-type TES \
    --lr 1e-2 --epoch 200 \
    --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 \
    --loss_align 1 --activate_regular_restrictions 3 --n_head 7 --hidden_dim 448 \
    --test --ckpt ./temp_log/fis-v/VST/cofinal/TES_best.pkl
