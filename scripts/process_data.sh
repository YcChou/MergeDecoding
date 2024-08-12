# Gsm8k
python -u ./preprocess.py \
    --data-path './data/Gsm8k' \
    --dataset 'Gsm8k' \

# StraQA
python -u ./preprocess.py \
    --data-path './data/StraQA' \
    --dataset 'StraQA' \

# WikiText
python -u ./preprocess.py \
    --data-path './data/WikiText' \
    --dataset 'WikiText' \
    --tokenizer {model-path}

# WikiNews
python -u ./preprocess.py \
    --data-path './data/WikiNews' \
    --dataset 'WikiNews' \
    --tokenizer {model-path}

# BookCorpus
python -u ./preprocess.py \
    --data-path './data/BookCorpus' \
    --dataset 'BookCorpus' \
    --tokenizer {model-path}