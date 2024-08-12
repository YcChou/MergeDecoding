from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
import torch
import argparse
from data import model
import test_loader
import pandas as pd
import tqdm
import os
import glob
import eval

class combine_vector:
    def __init__(self, pt_path):
        self.param = ''
        self.name = pt_path
        self._load()

    def _load(self):
        self.param = torch.load(self.name).half().cuda()

def test_single(args, combine_vector, LlmModel, Dataset):
    EvalClass = getattr(eval, f"eval_{args.dataset}")
    EvalClass(args, Dataset, LlmModel, combine_vector)

def main(args):
    # create dataloader
    LoaderClass = getattr(test_loader, f'{args.dataset}Loader')
    Dataset = LoaderClass(args)
    
    # load model
    ModelClass = getattr(model, args.model.replace('-', '_'))
    LlmModel = ModelClass(args)
    
    if args.vector_pts:
        if '.pt' in args.vector_pts:
            for pt in args.vector_pts.split(','):
                vector = combine_vector(pt)
                test_single(args, vector, LlmModel, Dataset)

        else: # folder
            args.test_folder = True
            pt_files = glob.glob(os.path.join(args.vector_pts, '*.pt'))
            for file_path in pt_files:
                pt = os.path.abspath(file_path)
                vector = combine_vector(pt)
                test_single(args, vector, LlmModel, Dataset)
                # file_name = os.path.basename(file_path)

    elif args.vector_pt:
        vector = combine_vector(args.vector_pt)
        test_single(args, vector, LlmModel, Dataset)

    else:
        # 'base' donot need vector
        test_single(args, None, LlmModel, Dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['Gsm8k', 'StraQA', 'WikiText', 'WikiNews', 'BookCorpus'], type=str, help='Name of the dataset loader to use')
    parser.add_argument('--model', required=True, choices=['llama1-7b', 'llama1-13b', 'llama1-30b'], type=str, help='Name of the model to use')
    parser.add_argument('--data-path', required=True, type=str, help='Path of valid or test dataset')
    parser.add_argument('--eval-type', required=True, type=str, choices=['base_gen', 'generate'], help='eval modes')
    parser.add_argument('--relative-top', default=True, help='if relative-top used')
    parser.add_argument('--result-path', required=True, type=str, help='Path of results')   
    parser.add_argument('--concerned-labels', default='0,2,4,6,8,10,12,14,16', type=str)
    parser.add_argument('--penalty', default=None, type=float)
    # vector
    parser.add_argument('--vector-pt', type=str, default='', help='vector pt after training')
    parser.add_argument('--vector-pts', type=str, default='', help='Test all vector in list')
    parser.add_argument('--test-folder', type=bool, default=False, help='test with vectors in the folder')
    parser.add_argument('--num-labels', default=17, type=int, help='eg. 17 classes for llama-7b')
    parser.add_argument('--max-len', default=512, type=int, help='max length for classifier')
    parser.add_argument('--wo-final', default=False, action="store_true")

    # special
    parser.add_argument('--task', default='', help='Assign specific tasks in MMLU dataset')
    args = parser.parse_args()

    main(args)
