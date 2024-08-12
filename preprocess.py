import argparse
import re
import json
import pandas as pd
from tqdm import tqdm 
import os
from sklearn.model_selection import train_test_split
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def prepare_gsm8k(args):
    print('Preparing data for Gsm8k...')
    train_path = os.path.join(args.data_path, 'train')
    valid_path = os.path.join(args.data_path, 'valid')
    test_path = os.path.join(args.data_path, 'test')

    if not os.path.exists(train_path):
        # split dataset
        for directory in [train_path, valid_path, test_path]:
            os.makedirs(directory, exist_ok=True)

        list_data_dict = []

        def clean(answer):
            answer = re.sub(r'<<.*?>>', '', answer)
            answer = answer.replace('\n', ' ')
            answer = answer.replace('####', 'The answer is')

            return answer + '.'
        
        def Gsm8k_create_demo_text(n_shot=8):
            question, chain, answer = [], [], []
            question.append("There are 15 trees in the grove. "
                            "Grove workers will plant trees in the grove today. "
                            "After they are done, there will be 21 trees. "
                            "How many trees did the grove workers plant today?")
            chain.append("There are 15 trees originally. "
                        "Then there were 21 trees after some more were planted. "
                        "So there must have been 21 - 15 = 6.")
            answer.append("6")

            question.append(
                "If there are 3 cars in the parking lot and 2 more cars arrive, "
                "how many cars are in the parking lot?")
            chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
            answer.append("5")

            question.append(
                "Leah had 32 chocolates and her sister had 42. If they ate 35, "
                "how many pieces do they have left in total?")
            chain.append("Originally, Leah had 32 chocolates. "
                        "Her sister had 42. So in total they had 32 + 42 = 74. "
                        "After eating 35, they had 74 - 35 = 39.")
            answer.append("39")

            question.append(
                "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
                "has 12 lollipops. How many lollipops did Jason give to Denny?")
            chain.append(
                "Jason started with 20 lollipops. Then he had 12 after giving some "
                "to Denny. So he gave Denny 20 - 12 = 8.")
            answer.append("8")

            question.append(
                "Shawn has five toys. For Christmas, he got two toys each from his "
                "mom and dad. How many toys does he have now?")
            chain.append(
                "Shawn started with 5 toys. If he got 2 toys each from his mom and "
                "dad, then that is 4 more toys. 5 + 4 = 9.")
            answer.append("9")

            question.append(
                "There were nine computers in the server room. Five more computers "
                "were installed each day, from monday to thursday. "
                "How many computers are now in the server room?")
            chain.append(
                "There were originally 9 computers. For each of 4 days, 5 more "
                "computers were added. So 5 * 4 = 20 computers were added. "
                "9 + 20 is 29.")
            answer.append("29")

            question.append(
                "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
                "wednesday, he lost 2 more. "
                "How many golf balls did he have at the end of wednesday?")
            chain.append(
                "Michael started with 58 golf balls. After losing 23 on tuesday, "
                "he had 58 - 23 = 35. After losing 2 more, "
                "he had 35 - 2 = 33 golf balls.")
            answer.append("33")

            question.append("Olivia has $23. She bought five bagels for $3 each. "
                            "How much money does she have left?")
            chain.append("Olivia had 23 dollars. "
                        "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
                        "So she has 23 - 15 dollars left. 23 - 15 is 8.")
            answer.append("8")

            index_list = list(range(len(question)))

            ANSWER_TRIGGER = "The answer is"
            # Concatenate demonstration examples ...
            demo_text = ""
            for i in index_list[:n_shot]:
                demo_text += "Q: " + question[i] + "\nA: " + chain[i] + " " + \
                            ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
                
            return demo_text

        shot = Gsm8k_create_demo_text(n_shot=8)
        with open(os.path.join(args.data_path, 'gsm8k_test.jsonl'), 'r') as f:
            for line in f:
                item = json.loads(line)
                new_item = dict(
                    question=shot + "Q: " + item['question'] + "\n" + "A:",
                    answer=clean(item['answer']))
                item = new_item
                list_data_dict.append(item)

        Dataset_df = pd.DataFrame(list_data_dict)
        train_valid_data, test_data = train_test_split(Dataset_df, test_size=0.1, random_state=42)
        train_data, valid_data = train_test_split(train_valid_data, test_size=0.2, random_state=42)
        train_data.to_csv(os.path.join(train_path, 'train.csv'), index=False)
        valid_data.to_csv(os.path.join(valid_path, 'valid.csv'), index=False)
        test_data.to_csv(os.path.join(test_path, 'test.csv'), index=False)
        print('Done.')
    else:
        print('File already exists.')
    return

def prepare_straqa(args):
    print('Preparing data for StraQA...')
    train_path = os.path.join(args.data_path, 'train')
    valid_path = os.path.join(args.data_path, 'valid')
    test_path = os.path.join(args.data_path, 'test')

    if not os.path.exists(train_path):
        # split dataset
        for directory in [train_path, valid_path, test_path]:
            os.makedirs(directory, exist_ok=True)

        list_data_dict = []

        def StraQA_create_demo_text(n_shot=6):
            QUESTION_TRIGGER = "\nLet's think step by step. "
            ANSWER_TRIGGER = "So the answer is"
            question, chain, answer = [], [], []

            question.append("Do hamsters provide food for any animals?")
            chain.append("Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals.")
            answer.append("yes")

            question.append("Could Brooke Shields succeed at University of Pennsylvania?")
            chain.append("Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania.")
            answer.append("yes")

            question.append("Hydrogen's atomic number squared exceeds number of Spice Girls?")
            chain.append("Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen's atomic number squared is less than 5.")
            answer.append("no")

            question.append("Is it common to see frost during some college commencements?")
            chain.append("College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements.")
            answer.append("yes")
            
            question.append("Could a llama birth twice during War in Vietnam (1945-46)?")
            chain.append("The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam.")
            answer.append("no")

            question.append("Would a pear sink in water?")
            chain.append("The density of a pear is about 0.59 g/cm^3, which is less than water. Objects less dense than water float. Thus, a pear would float.")
            answer.append("no")

            # randomize order of the examples ...
            demo_text = ''
            index_list = list(range(len(question)))

            for i in index_list[:n_shot]:
                demo_text += "Q: " + question[i] + "\nA: " + chain[i] + " " + ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
                
            return demo_text

        shot = StraQA_create_demo_text(n_shot=6)
        with open(os.path.join(args.data_path, 'strategyqa_train.json'), 'r') as f:
            items = json.load(f)
            for item in items:
                question = item.get('question', None)
                answer = 'yes' if item.get('answer', None) else 'no'
                facts = '. '.join([fact.strip()for fact in item.get('facts', [])])

                new_item = dict(
                    question = shot + "Q: " + question + "\n" + "A:",
                    answer = facts + " So the answer is " + answer)
                item = new_item
                list_data_dict.append(item)

        Dataset_df = pd.DataFrame(list_data_dict)
        train_valid_data, test_data = train_test_split(Dataset_df, test_size=0.1, random_state=42)
        train_data, valid_data = train_test_split(train_valid_data, test_size=0.2, random_state=42)
        train_data.to_csv(os.path.join(train_path, 'train.csv'), index=False)
        valid_data.to_csv(os.path.join(valid_path, 'valid.csv'), index=False)
        test_data.to_csv(os.path.join(test_path, 'test.csv'), index=False)
        print('Done.')
    else:
        print('File already exists.')
    return

def prepare_wikitext(args):
    print('Preparing data for StraQA...')
    train_path = os.path.join(args.data_path, 'train')
    valid_path = os.path.join(args.data_path, 'valid')
    test_path = os.path.join(args.data_path, 'test')

    if not os.path.exists(train_path):
        # split dataset
        for directory in [train_path, valid_path, test_path]:
            os.makedirs(directory, exist_ok=True)

        list_data_dict = []

        dataset_size = 2000
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer, use_fast=False, padding_side="right")
        datasets = load_dataset(data_files={'val':['./data/WikiText/wiki.test.raw', './data/WikiText/wiki.valid.raw']}, path="text")

        list_data_dict = []

        count = 0
        for i in datasets['val']:
            text_all = i['text']
            item_ids = tokenizer(text_all, return_tensors="pt").input_ids.to('cuda')

            if len(item_ids[0]) >= 160:
                item = {
                    'question': tokenizer.decode(item_ids[0][:32], skip_special_tokens=True),
                    'answer': tokenizer.decode(item_ids[0][32:], skip_special_tokens=True),
                }
                list_data_dict.append(item)
                count += 1

            if count == dataset_size:
                break

        Dataset_df = pd.DataFrame(list_data_dict)
        train_valid_data, test_data = train_test_split(Dataset_df, test_size=0.1, random_state=42)
        train_data, valid_data = train_test_split(train_valid_data, test_size=0.2, random_state=42)
        train_data.to_csv(os.path.join(train_path, 'train.csv'), index=False)
        valid_data.to_csv(os.path.join(valid_path, 'valid.csv'), index=False)
        test_data.to_csv(os.path.join(test_path, 'test.csv'), index=False)
        print('Done.')
    else:
        print('File already exists.')
    return

def prepare_wikinews(args):
    print('Preparing data for WikiNews...')
    train_path = os.path.join(args.data_path, 'train')
    valid_path = os.path.join(args.data_path, 'valid')
    test_path = os.path.join(args.data_path, 'test')

    if not os.path.exists(train_path):
        # split dataset
        for directory in [train_path, valid_path, test_path]:
            os.makedirs(directory, exist_ok=True)

        list_data_dict = []

        dataset_size = 2000
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer, use_fast=False, padding_side="right")
        list_data_dict = []

        count = 0
        with open(os.path.join(args.data_path, 'megarhyme-wikinews.json'), 'r') as f:
            items = json.load(f)
            for item in items:
                text_all = item['text']
                item_ids = tokenizer(text_all, return_tensors="pt").input_ids.to('cuda')

                if 500 >= len(item_ids[0]) >= 160:
                    item = {
                        'question': tokenizer.decode(item_ids[0][:32], skip_special_tokens=True),
                        'answer': tokenizer.decode(item_ids[0][32:], skip_special_tokens=True),
                    }
                    list_data_dict.append(item)
                    count += 1

                if count == dataset_size:
                    print(f'Dataset of size {dataset_size} is generated.')
                    break

        Dataset_df = pd.DataFrame(list_data_dict)
        train_valid_data, test_data = train_test_split(Dataset_df, test_size=0.1, random_state=42)
        train_data, valid_data = train_test_split(train_valid_data, test_size=0.2, random_state=42)
        train_data.to_csv(os.path.join(train_path, 'train.csv'), index=False)
        valid_data.to_csv(os.path.join(valid_path, 'valid.csv'), index=False)
        test_data.to_csv(os.path.join(test_path, 'test.csv'), index=False)
        print('Done.')
    else:
        print('File already exists.')
    return

def prepare_bookcorpus(args):
    print('Preparing data for BookCorpus...')
    train_path = os.path.join(args.data_path, 'train')
    valid_path = os.path.join(args.data_path, 'valid')
    test_path = os.path.join(args.data_path, 'test')

    if not os.path.exists(train_path):
        # split dataset
        for directory in [train_path, valid_path, test_path]:
            os.makedirs(directory, exist_ok=True)

        list_data_dict = []

        dataset_size = 2000
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer, use_fast=False, padding_side="right")
        list_data_dict = []

        count = 0
        dataset = load_dataset("bookcorpus")

        index = len(dataset['train']) // 2
        while index < len(dataset['train']):
            # import pdb; pdb.set_trace()
            text_all = ''
            for i in range(10):
                text_all += dataset['train'][index + i]['text']

            item_ids = tokenizer(text_all, return_tensors="pt").input_ids.to('cuda')
            item = {
                'question': tokenizer.decode(item_ids[0][:32], skip_special_tokens=True),
                'answer': tokenizer.decode(item_ids[0][32:], skip_special_tokens=True),
            }
            list_data_dict.append(item)
            count += 1
            index += 10

            if count == dataset_size:
                print(f'Dataset of size {dataset_size} is generated.')
                break

        Dataset_df = pd.DataFrame(list_data_dict)
        train_valid_data, test_data = train_test_split(Dataset_df, test_size=0.1, random_state=42)
        train_data, valid_data = train_test_split(train_valid_data, test_size=0.2, random_state=42)
        train_data.to_csv(os.path.join(train_path, 'train.csv'), index=False)
        valid_data.to_csv(os.path.join(valid_path, 'valid.csv'), index=False)
        test_data.to_csv(os.path.join(test_path, 'test.csv'), index=False)
        print('Done.')
    else:
        print('File already exists.')
    return



def main(args):
    if args.dataset == 'Gsm8k':
        prepare_gsm8k(args)
    elif args.dataset == 'StraQA':
        prepare_straqa(args)
    elif args.dataset == 'WikiText':
        prepare_wikitext(args)
    elif args.dataset == 'WikiNews':
        prepare_wikinews(args)
    elif args.dataset == 'BookCorpus':
        prepare_bookcorpus(args)

def parse_args():
    parser = argparse.ArgumentParser(description='prepare for training')
    parser.add_argument('--data-path', default='', type=str, help='training data and validing data')
    parser.add_argument('--dataset', required=True, choices=['Gsm8k', 'StraQA', 'WikiText', 'WikiNews', 'BookCorpus'], type=str, help='Name of the dataset loader to use')
    parser.add_argument('--tokenizer', type=str, help='used for open-ended generation task')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)