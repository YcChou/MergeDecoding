export CUDA_VISIBLE_DEVICES=2

model='llama1-7b'
# model='llama1-13b'
# model='llama1-30b'

declare -A label_map
label_map['llama1-7b']=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
label_map['llama1-13b']=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
label_map['llama1-30b']=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30

lr=0.005

python -u ./train.py \
    --model ${model} \
    --data-path ./data/bookcorpus \
    --save-path ./ckpt/${model}/bookcorpus \
    --epoch 30 \
    --batch-size 16 \
    --print-every 50 \
    --save-every 50 \
    --lr ${lr} \
    --concerned-labels ${label_map[$model]}\

python -u ./test.py \
    --dataset 'BookCorpus' \
    --model ${model} \
    --data-path ./data/bookcorpus \
    --vector-pts ./ckpt/${model}/bookcorpus/lr-epoch-bs-${lr}-30-16 \
    --result-path ./ckpt/${model}/bookcorpus/lr-epoch-bs-${lr}-30-16 \
    --eval-type 'generate' \
    --concerned-labels ${label_map[$model]}\
    --max-len 128