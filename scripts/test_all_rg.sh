path='/home/zkl/Documents/Data/GDLT_data/VST/RG'
gpu='0'

CUDA_VISIBLE_DEVICES=${gpu} python main.py \
    --video-path ${path}'/swintx_avg_fps25_clip32' \
    --train-label-path ${path}'/train.txt' \
    --test-label-path ${path}'/test.txt' \
    --model-name Ball \
    --submodel-name cofinal \
    --action-type Ball \
    --lr 1e-2 --epoch 250 \
    --n_decoder 2 --n_query 4 --alpha 1 --margin 1 --lr-decay cos --decay-rate 0.01 --dropout 0.3 \
    --loss_align 1 --activate_regular_restrictions 2 \
    --test --ckpt ./temp_log/RG/VST/cofinal/Ball_best.pkl

CUDA_VISIBLE_DEVICES=${gpu} python main.py \
    --video-path ${path}'/swintx_avg_fps25_clip32' \
    --train-label-path ${path}'/train.txt' \
    --test-label-path ${path}'/test.txt' \
    --model-name Clubs \
    --submodel-name cofinal \
    --action-type Clubs \
    --lr 1e-2 --epoch 400 \
    --n_decoder 2 --n_query 4 --alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 \
    --loss_align 1 --activate_regular_restrictions 3 \
    --test --ckpt ./temp_log/RG/VST/cofinal/Clubs_best.pkl

CUDA_VISIBLE_DEVICES=${gpu} python main.py \
    --video-path ${path}'/swintx_avg_fps25_clip32' \
    --train-label-path ${path}'/train.txt' \
    --test-label-path ${path}'/test.txt' \
    --model-name Hoop \
    --submodel-name cofinal \
    --action-type Hoop \
    --lr 1e-2 --epoch 300 \
    --n_decoder 2 --n_query 4 --alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 \
    --loss_align 2 --activate_regular_restrictions 2 \
    --test --ckpt ./temp_log/RG/VST/cofinal/Hoop_best.pkl

CUDA_VISIBLE_DEVICES=${gpu} python main.py \
    --video-path ${path}'/swintx_avg_fps25_clip32' \
    --train-label-path ${path}'/train.txt' \
    --test-label-path ${path}'/test.txt' \
    --model-name Ribbon \
    --submodel-name cofinal_2 \
    --action-type Ribbon \
    --lr 1e-2 --epoch 150 \
    --n_decoder 2 --n_query 4 --alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 \
    --loss_align 1 --activate_regular_restrictions 3 \
    --test --ckpt ./temp_log/RG/VST/cofinal_2/Ribbon_best.pkl