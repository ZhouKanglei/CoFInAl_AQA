CUDA_VISIBLE_DEVICES="2" python main.py \
    --video-path '/home/zkl/Documents/Data/GDLT_data/swintx_avg_fps25_clip32' \
    --train-label-path '/home/zkl/Documents/Data/GDLT_data/train.txt' \
    --test-label-path '/home/zkl/Documents/Data/GDLT_data/test.txt' \
    --model-name Ball \
    --submodel-name GDLTETH1 \
    --action-type Ball \
    --lr 1e-2 --epoch 250 \
    --n_decoder 2 --n_query 4 --alpha 1 --margin 1 --lr-decay cos --decay-rate 0.01 --dropout 0.3 \
    --loss_align 1 --activate_regular_restrictions 2

python main.py \
    --video-path '/home/zkl/Documents/Data/GDLT_data/swintx_avg_fps25_clip32' \
    --train-label-path '/home/zkl/Documents/Data/GDLT_data/train.txt' \
    --test-label-path '/home/zkl/Documents/Data/GDLT_data/test.txt' \
    --model-name Clubs \
    --submodel-name GDLTETH1 \
    --action-type Clubs\
    --lr 1e-2 --epoch 400 \
    --n_decoder 2 --n_query 4 --alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 \
    --loss_align 1 --activate_regular_restrictions 3

python main.py \
    --video-path '/home/zkl/Documents/Data/GDLT_data/swintx_avg_fps25_clip32' \
    --train-label-path '/home/zkl/Documents/Data/GDLT_data/train.txt' \
    --test-label-path '/home/zkl/Documents/Data/GDLT_data/test.txt' \
    --model-name Hoop \
    --submodel-name GDLTETH1 \
    --action-type Hoop\
    --lr 1e-2 --epoch 500 \
    --n_decoder 2 --n_query 4 --alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 \
    --loss_align 2 --activate_regular_restrictions 2

python main.py \
    --video-path '/home/zkl/Documents/Data/GDLT_data/swintx_avg_fps25_clip32' \
    --train-label-path '/home/zkl/Documents/Data/GDLT_data/train.txt' \
    --test-label-path '/home/zkl/Documents/Data/GDLT_data/test.txt' \
    --model-name Ribbon \
    --submodel-name GDLTETH2 \
    --action-type Ribbon \
    --lr 1e-2 --epoch 150 \
    --n_decoder 2 --n_query 4 --alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 \
    --loss_align 1 --activate_regular_restrictions 3


python main.py \
    --video-path '/home/zkl/Documents/Data/GDLT_data/swintx_avg_fps25_clip32' \
    --train-label-path '/home/zkl/Documents/Data/GDLT_data/train.txt' \
    --test-label-path '/home/zkl/Documents/Data/GDLT_data/test.txt' \
    --model-name TES \
    --submodel-name GDLTETH1 \
    --action-type TES\
    --lr 1e-2 --epoch