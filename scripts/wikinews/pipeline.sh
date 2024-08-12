export CUDA_VISIBLE_DEVICES=0

model='llama1-7b'
# model='llama1-13b'
# model='llama1-30b'
# model='llama1-65b'

declare -A label_map
label_map['llama1-7b']=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
label_map['llama1-13b']=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
label_map['llama1-30b']=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30

lr=0.01

python -u ./train.py \
    --model ${model} \
    --data-path ./data/wikinews \
    --save-path ./ckpt/${model}/wikinews \
    --epoch 30 \
    --batch-size 16 \
    --print-every 50 \
    --save-every 50 \
    --lr ${lr} \
    --concerned-labels ${label_map[$model]}\

python -u ./test.py \
    --dataset 'WikiNews' \
    --model ${model} \
    --data-path ./data/wikinews \
    --vector-pts ./ckpt/${model}/wikinews/lr-epoch-bs-${lr}-30-16 \
    --result-path ./ckpt/${model}/wikinews/lr-epoch-bs-${lr}-30-16 \
    --eval-type 'generate' \
    --concerned-labels ${label_map[$model]}\
    --max-len 128