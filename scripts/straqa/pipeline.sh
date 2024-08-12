export CUDA_VISIBLE_DEVICES=0

model='llama1-7b'
# model='llama1-13b'
# model='llama1-30b'
# model='llama1-65b'

declare -A label_map
label_map['llama1-7b']=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
label_map['llama1-13b']=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
label_map['llama1-30b']=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30

lr=1e-05

python -u ./train.py \
    --model ${model} \
    --data-path ./data/StraQA \
    --save-path ./ckpt/${model}/StraQA \
    --epoch 20 \
    --batch-size 32 \
    --lr ${lr} \
    --print-every 20 \
    --save-every 20 \
    --concerned-labels ${label_map[$model]}\

python -u ./test.py \
    --dataset 'StraQA' \
    --model ${model} \
    --data-path ./data/StraQA \
    --vector-pts ./ckpt/${model}/StraQA/lr-epoch-bs-${lr}-20-32 \
    --result-path ./ckpt/${model}/StraQA/lr-epoch-bs-${lr}-20-32 \
    --eval-type 'generate' \
    --concerned-labels ${label_map[$model]}\
