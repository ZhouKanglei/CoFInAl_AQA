sub_model_name="GDLTETH1" # GDLTETH1/GDLTETH2/HGCN/GDLT
python main.py --video-path Path/To/DataGDLT_data/swintx_avg_fps25_clip32 \
    --train-label-path Path/To/DataGDLT_data/train.txt \
    --test-label-path Path/To/DataGDLT_data/test.txt  \
    --n_decoder 2 --n_query 4 --dropout 0.3 --test \
    --model-name Ball \
    --submodel-name ${sub_model_name} \
    --action-type Ball \
    --ckpt Path/To/CKPT/temp_log/${sub_model_name}/Ball_best.pkl

python main.py --video-path Path/To/DataGDLT_data/swintx_avg_fps25_clip32 \
    --train-label-path Path/To/DataGDLT_data/train.txt \
    --test-label-path Path/To/DataGDLT_data/test.txt  \
    --n_decoder 2 --n_query 4 --dropout 0.3 --test \
    --model-name Clubs \
    --submodel-name ${sub_model_name} \
    --action-type Clubs \
    --ckpt Path/To/CKPT/temp_log/${sub_model_name}/Clubs_best.pkl

python main.py --video-path Path/To/DataGDLT_data/swintx_avg_fps25_clip32 \
    --train-label-path Path/To/DataGDLT_data/train.txt \
    --test-label-path Path/To/DataGDLT_data/test.txt  \
    --n_decoder 2 --n_query 4 --dropout 0.3 --test \
    --model-name Hoop \
    --submodel-name ${sub_model_name} \
    --action-type Hoop \
    --ckpt Path/To/CKPT/temp_log/${sub_model_name}/Hoop_best.pkl

python main.py --video-path Path/To/DataGDLT_data/swintx_avg_fps25_clip32 \
    --train-label-path Path/To/DataGDLT_data/train.txt \
    --test-label-path Path/To/DataGDLT_data/test.txt  \
    --n_decoder 2 --n_query 4 --dropout 0.3 --test \
    --model-name Ribbon \
    --submodel-name GDLTETH2\
    --action-type Ribbon \
    --ckpt Path/To/CKPT/temp_log/GDLTETH2/Ribbon_best.pkl

python main.py --video-path Path/To/Datafis-v/swintx_avg_fps25_clip32 \
    --train-label-path Path/To/Datafis-v/train.txt \
    --test-label-path Path/To/Datafis-v/test.txt  \
    --n_decoder 2 --n_query 4 --dropout 0.3 --test \
    --model-name TES \
    --submodel-name ${sub_model_name} \
    --action-type TES \
    --ckpt Path/To/CKPT/temp_log/${sub_model_name}/TES_best.pkl

python main.py --video-path Path/To/Datafis-v/swintx_avg_fps25_clip32 \
    --train-label-path Path/To/Datafis-v/train.txt \
    --test-label-path Path/To/Datafis-v/test.txt  \
    --n_decoder 2 --n_query 4 --dropout 0.3 --test \
    --model-name PCS \
    --submodel-name ${sub_model_name} \
    --action-type PCS \
    --ckpt Path/To/CKPT/temp_log/${sub_model_name}/PCS_best.pkl