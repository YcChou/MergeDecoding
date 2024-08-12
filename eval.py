import eval_type
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import json
import os
import re
from simcse import SimCSE

def eval_Gsm8k(args, Dataset, LlmModel, vector):
    eval_mode = getattr(eval_type, args.eval_type)
    if args.test_folder:
        result_path = os.path.join(args.result_path, os.path.basename(vector.name).replace('.pt', ''))
    else:
        result_path = args.result_path

    os.makedirs(result_path, exist_ok=True)
    f = open(os.path.join(result_path, 'result.txt'), 'w')

    if args.eval_type == 'base_gen':
        print(f'Generate Task: Eval {args.dataset} with {args.model} without vector')  
    else:
        print(f'Generate Task: Eval {args.dataset} with {args.model} and vector in {vector.name}')
    
    def Gsm8k_clean_answer(model_pred):
        ANSWER_TRIGGER = "The answer is"

        model_pred = model_pred.lower()
        preds = model_pred.split(ANSWER_TRIGGER.lower())
        answer_flag = True if len(preds) > 1 else False
        if answer_flag:
            # Pick first answer with flag
            pred = preds[1]
        else:
            # Pick last number without flag
            pred = preds[-1]

        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

        if len(pred) == 0:
            return "[invalid]"
        
        if answer_flag: # choose the first element in list
            pred = pred[0]
        else: # choose the last element in list
            pred = pred[-1]

        # (For arithmetic tasks) if a word ends with period, it will be omitted ...
        if pred[-1] == ".":
            pred = pred[:-1]

        return pred
    
    def Gsm8k_is_correct(model_answer, answer):
        assert answer != "[invalid]"
        return model_answer == answer
    
    answers = []
    for sample in tqdm(Dataset):
        model_completion = eval_mode(args, LlmModel, sample['question'], vector)
        model_completion = model_completion.strip()

        model_answer = Gsm8k_clean_answer(model_completion)
        is_cor = Gsm8k_is_correct(model_answer, sample['answer'])
        answers.append(is_cor)

        print(f'Question: {sample["question"]}\n\n'
            f'Answers: {sample["answer"]}\n\n'
            f'Model Answers: {model_answer}\n\n'
            f'Model Completion: {model_completion}\n\n'
            f'Is correct: {is_cor}\n\n', file=f)

        print(f'Num of total question: {len(answers)}, '
            f'correct num: {sum(answers)}, '
            f'correct rate: {float(sum(answers))/len(answers)}.', file=f)

    print(f"Final acc: {float(sum(answers))/len(answers)}", file=f)
    f.close()
    return

def eval_StraQA(args, Dataset, LlmModel, vector):
    eval_mode = getattr(eval_type, args.eval_type)
    if args.test_folder:
        result_path = os.path.join(args.result_path, os.path.basename(vector.name).replace('.pt', ''))
    else:
        result_path = args.result_path

    os.makedirs(result_path, exist_ok=True)
    f = open(os.path.join(result_path, 'result.txt'), 'w')

    if args.eval_type == 'base_gen':
        print(f'Generate Task: Eval {args.dataset} with {args.model} without vector')  
    else:
        print(f'Generate Task: Eval {args.dataset} with {args.model} and vector in {vector.name}')

    def StraQA_clean_answer(model_pred):
        SHORT_ANSWER_TRIGGER = "answer is"
        model_pred = model_pred.lower()

        if "Thus, yes." in model_pred:
            preds = "yes"
        elif SHORT_ANSWER_TRIGGER.lower() in model_pred:
            preds = model_pred.split(SHORT_ANSWER_TRIGGER.lower())[1].split(".")[0].strip()
        else:
            print("Warning: answer trigger not found in model prediction:", model_pred, "; returning yes/no based on exact match of `no`.", flush=True)
            preds = "no" if "no" in model_pred else "yes"

        if preds not in ["yes", "no"]:
            print("Warning: model prediction is not yes/no:", preds, "; returning no", flush=True)
            preds = "no"

        return (preds == "yes")
    
    def StraQA_is_correct(model_answer, answer):
        gt_answer = answer
        assert gt_answer != "[invalid]"
        return model_answer == gt_answer

    answers = []
    for sample in tqdm(Dataset):
        model_completion = eval_mode(args, LlmModel, sample['question'], vector)
        model_completion = model_completion.strip()

        model_answer = StraQA_clean_answer(model_completion)
        is_cor = StraQA_is_correct(model_answer, sample['answer'])
        answers.append(is_cor)

        print(f'Question: {sample["question"]}\n\n'
            f'Answers: {sample["answer"]}\n\n'
            f'Model Answers: {model_answer}\n\n'
            f'Model Completion: {model_completion}\n\n'
            f'Is correct: {is_cor}\n\n', file=f)

        print(f'Num of total question: {len(answers)}, '
            f'correct num: {sum(answers)}, '
            f'correct rate: {float(sum(answers))/len(answers)}.', file=f)

    print(f"Final acc: {float(sum(answers))/len(answers)}", file=f)
    f.close()
    return

def eval_WikiText(args, Dataset, LlmModel, vector):
    eval_mode = getattr(eval_type, args.eval_type)
    if args.test_folder:
        result_path = os.path.join(args.result_path, os.path.basename(vector.name).replace('.pt', ''))
    else:
        result_path = args.result_path

    os.makedirs(result_path, exist_ok=True)

    if args.eval_type == 'base_gen':
        print(f'Generate Task: Eval {args.dataset} with {args.model} without vector')  
    else:
        print(f'Generate Task: Eval {args.dataset} with {args.model} and vector in {vector.name}')

    if not os.path.exists(os.path.join(result_path, 'generate.csv')):
        result_dict = {'prompt': [], 'generated_woprompt': [], 'label_woprompt': []}
        for sample in tqdm(Dataset):
            generated_woprompt = eval_mode(args, LlmModel, sample['question'], vector)
            generated_woprompt = generated_woprompt.strip()
            label_woprompt = sample['answer']

            result_dict['prompt'].append(sample['question'])
            result_dict['generated_woprompt'].append(generated_woprompt)
            result_dict['label_woprompt'].append(label_woprompt)

        results_df = pd.DataFrame(result_dict)
        results_df.to_csv(os.path.join(result_path, 'generate.csv'), index=False)
    
    # eval generated text
    metric_dic = {}
    df = pd.read_csv(os.path.join(result_path, 'generate.csv'))
    label_woprompt = df['label_woprompt'].to_numpy()
    generated_woprompt = df['generated_woprompt'].to_numpy()

    coh_model = SimCSE('princeton-nlp--unsup-simcse-bert-base-uncased')
    metric_dic['ppl'] = ppl(LlmModel.model, LlmModel.tokenizer, generated_woprompt)
    metric_dic['diversity'] = diversity(generated_woprompt)
    metric_dic['mauve'] = mauve(generated_woprompt, label_woprompt)
    metric_dic['coherence'] = coherence(coh_model, generated_woprompt, label_woprompt)
    metric_dic['flesch'] = flesch(generated_woprompt)

    # import pdb; pdb.set_trace()
    with open(os.path.join(result_path, "results.txt"), "w") as file:
        args_dict = vars(args)
        args_str = json.dumps(args_dict, indent=4)
        file.write(args_str)
        file.write('\n')

        for metric, value in metric_dic.items():
            file.write(f"{metric:35}: {value:6}\n")
    
    return

def eval_WikiNews(args, Dataset, LlmModel, vector):
    eval_WikiText(args, Dataset, LlmModel, vector)

def eval_BookCorpus(args, Dataset, LlmModel, vector):
    eval_WikiText(args, Dataset, LlmModel, vector)

# Metrics
def clean(lists):
    cleaned = []
    for item in lists:
        item_list = item.split(' ')
        cleaned.append(' '.join(item_list))
    return cleaned

def diversity(text_list):
    '''
    text_list: the list of text
    '''
    text_list = clean(text_list)
    def eval_text(text, ngram):
        token_list = text.strip().split()
        start_idx, end_idx = 0, ngram
        total_num = 0
        ngram_set = set()
        while end_idx < len(token_list):
            one_ngram_list = token_list[start_idx:end_idx]
            assert len(one_ngram_list) == ngram
            one_ngram = ' '.join(one_ngram_list)
            total_num += 1
            ngram_set.add(one_ngram)
            start_idx += 1
            end_idx += 1
        return len(ngram_set), total_num

    def eval_one_instance(text, ngram_list):
        res_dict = {}
        for n in ngram_list:
            n_unique, n_total = eval_text(text, n)
            res_dict[n] = {'unique':n_unique, 'total':n_total}
        unique_token_set = set(text.strip().split())
        return res_dict, unique_token_set

    ngram_list = [2,3,4]
    pred_res_dict = {}
    for n in ngram_list:
        pred_res_dict[n] = {}
        pred_res_dict[n]['unique'] = 0
        pred_res_dict[n]['total'] = 0
    
    pred_unique_token_set = set()
    for text in text_list:
        text = text.strip('\n').strip()
        one_pred_res_dict, one_pred_uni_token_set = eval_one_instance(text, ngram_list)

        # unique token set
        pred_unique_token_set = pred_unique_token_set.union(one_pred_uni_token_set)
        # ngram statistic
        for n in ngram_list:
            pred_res_dict[n]['unique'] += one_pred_res_dict[n]['unique']
            pred_res_dict[n]['total'] += one_pred_res_dict[n]['total']

    # prediction result
    pred_seq_2 = 1 - (pred_res_dict[2]['unique']/pred_res_dict[2]['total'])
    pred_seq_2 = round(pred_seq_2 * 100, 2)
    pred_seq_3 = 1 - (pred_res_dict[3]['unique']/pred_res_dict[3]['total'])
    pred_seq_3 = round(pred_seq_3 * 100, 2)
    pred_seq_4 = 1 - (pred_res_dict[4]['unique']/pred_res_dict[4]['total'])
    pred_seq_4 = round(pred_seq_4 * 100, 2)
    pred_div = (1 - pred_seq_2/100) * (1 - pred_seq_3/100) * (1 - pred_seq_4/100)

    # print('Repetition and diversity:')
    # print ('rep-2 is {}, rep-3 is {}, rep-4 is {}, and diversity is {}.\n'.format(pred_seq_2, pred_seq_3, pred_seq_4, round(pred_div,5)))
    return pred_div

def ppl(model, tokenizer, generated_text):
    generated_text = clean(generated_text)

    bsz_size = 20 
    score_lst = []
    for i in tqdm(range(len(generated_text)//bsz_size)):
        text_list_i = generated_text[i*bsz_size:(i+1) * bsz_size]
        inputs = tokenizer(text_list_i, return_tensors='pt', padding=True)
        with torch.no_grad():
            labels = inputs['input_ids'].cuda() 
            labels[labels== tokenizer.pad_token] = -100 
            out = model(input_ids=inputs['input_ids'].cuda(), attention_mask=inputs['attention_mask'].cuda(), labels=labels)
            # print(out.loss) 
            score_lst.append(out.loss) 
    score_lst = torch.tensor(score_lst)
    ppl = np.e ** score_lst.mean()

    print('Perplexity: {}'.format(round(ppl.item(),5)))
    return ppl.item()

def mauve(generated_woprompt, label_woprompt):
    '''
    https://github.com/krishnap25/mauve
    '''
    import mauve
    generated_woprompt = clean(generated_woprompt)
    label_woprompt = clean(label_woprompt)

    out = mauve.compute_mauve(p_text=label_woprompt, q_text=generated_woprompt, device_id=0, max_text_length=256, verbose=False, featurize_model_name='bert-base-uncased', batch_size=20)
    # print(f'mauve: {out.mauve}')
    return out.mauve

def coherence(model, pp_lst, yy_lst):

    full_sim_lst = []
    # pp_lst, yy_lst = zip(*sent_lst)
    # pp_lst = list(pp_lst)
    # yy_lst = list(yy_lst) 
    # print(len(pp_lst), len(yy_lst))
    pp_lst = clean(pp_lst)
    yy_lst = clean(yy_lst)

    similarities = model.similarity(pp_lst, yy_lst)
    similarities = np.array(similarities)
    coherence_score = similarities.trace() / len(similarities) 
    # print(f'coherence: {round(coherence_score, 2)}')
    return round(coherence_score, 2)
    
def flesch(generated_woprompt):
    import re
    import pronouncing
    generated_woprompt = clean(generated_woprompt)
    def word_list(sentence):
        word_re = re.compile(r'[^A-Za-z’\']+')
        words = word_re.split(sentence.lower())

        return words

    def sentence_count(sentence):
        if sentence[-1] != '.':
            sentence = sentence + '.'

        point_re = re.compile(r'\.')
        point = point_re.split(sentence)

        return (len(point)-1)

    def get_pronouncing_num(word):
        pronunciation_list = pronouncing.phones_for_word(word)
        if pronunciation_list:
            num = pronouncing.syllable_count(pronunciation_list[0])
        else:
            return 0
        return num

    def get_pronouncing_nums(words):
        counts = 0
        for word in words:
            counts += get_pronouncing_num(word)

        return counts

    result_dict = {'Total_ASW':0, 'Total_ASL':0, 'Total_RE':0}
    item_num = len(generated_woprompt)

    for item in generated_woprompt:
        # ASL
        sentence = item.strip()
        words = word_list(sentence)
        word_num = len(words)
        sentence_num = sentence_count(sentence)
        ASL = word_num / sentence_num

        # ASW  
        pronouncing_nums = get_pronouncing_nums(words)
        ASW = pronouncing_nums / word_num

        # RE
        RE = 206.835 - (1.015 * ASL) - (84.6 * ASW)
        result_dict['Total_ASW'] += ASW
        result_dict['Total_ASL'] += ASL
        result_dict['Total_RE'] += RE
    RE = result_dict['Total_RE'] / item_num
    ASW = result_dict['Total_ASW'] / item_num
    ASL = result_dict['Total_ASL'] / item_num
    # print(f'flesch: RE={RE} (ASW={ASW}, ASL={ASL})')
    return RE
