

CUDA_VISIBLE_DEVICES="0" python main.py \
    --video-path '/home/zkl/Documents/Data/GDLT_data/VST/RG/swintx_avg_fps25_clip32' \
    --train-label-path '/home/zkl/Documents/Data/GDLT_data/VST/RG/train.txt' \
    --test-label-path '/home/zkl/Documents/Data/GDLT_data/VST/RG/test.txt' \
    --model-name Ball \
    --submodel-name cofinal \
    --action-type Ball \
    --lr 1e-2 --epoch 250 \
    --n_decoder 2 --n_query 4 --alpha 1 --margin 1 --lr-decay cos --decay-rate 0.01 --dropout 0.3 \
    --loss_align 1 --activate_regular_restrictions 2