from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
import argparse
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
import torch
from tqdm import tqdm
import os
from collections import Counter
import numpy as np
import random
from data import model as Model
import re
import json


def dataloader(args, tokenizer, flag):
    data_files = {}
    data_files["train"] = os.path.join(args.data_path, 'train/train.csv')
    data_files["valid"] = os.path.join(args.data_path, 'valid/valid.csv')
    raw_dataset = load_dataset('csv', data_files=data_files)

    def collate_fn(data):

        question = str(data[0]['question'])
        answer = str(data[0]['answer'])

        inputs = {}
        input_text = question + answer
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to('cuda')
        question_ids = tokenizer(question, return_tensors="pt").input_ids.to('cuda')
        label_ids = input_ids[0, question_ids.shape[-1]:]

        inputs['input_ids'] = input_ids
        inputs['question_ids'] = question_ids
        inputs['label_ids'] = label_ids
        
        inputs['input_text'] = input_text
        inputs['question'] = question
        inputs['answer'] = answer
        
        return inputs

    dataset = raw_dataset[flag]
    dataloader = DataLoader(dataset, shuffle=True, collate_fn=collate_fn, batch_size=1)
    
    return dataloader


def save_and_valid(args, step, combination_vector, model, ValidLoader):
    # save
    template = 'lr-epoch-bs-{:}-{:}-{:}'
    current_dict = os.path.join(args.save_path, template.format(args.lr, args.epoch, args.batch_size))

    if not os.path.exists(current_dict):
        os.makedirs(current_dict)

    torch.save(combination_vector, os.path.join(current_dict, f"{str(step)}.pt"))

    # valid
    print('\nValiding...')
    model.eval()

    total_loss = 0

    with torch.no_grad():
        total_loss = 0
        loss_fn = torch.nn.CrossEntropyLoss()

        for batch in ValidLoader:
            outputs = model(
                        input_ids=batch['input_ids'],
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True
                    )

            final_logits = combine(args, model, outputs.hidden_states, batch['question_ids'], combination_vector)
            final_logits = final_logits.to(batch['label_ids'].device)
            loss = loss_fn(final_logits, batch['label_ids'])

            total_loss += loss
    
    print('\n' + '-' * 20)
    print('Valid set: ')
    print(f'step={step}')
    print('loss: %.2f' % (total_loss/len(ValidLoader)))
    print('-' * 20)

def combine(args, model, logits, question_ids, combination_vector):
    mature_layer = len(logits)-1
    head_layer = model.get_output_embeddings()

    layer_dict = {int(i):0 for i in args.concerned_labels.split(',')}

    for premature_layer in layer_dict.keys():
        layer_dict[premature_layer] = head_layer(logits[premature_layer])[0, question_ids.shape[-1] - 1: -1, :].log_softmax(dim=-1)

    if not args.wo_final:
        layer_dict[mature_layer] = head_layer(logits[mature_layer])[0, question_ids.shape[-1] - 1: -1, :].log_softmax(dim=-1) # 加了log_softmax

    layer_list = [layer_dict[layer] for layer in layer_dict.keys()]
    combination_vector = combination_vector.unsqueeze(1)  # (num_labels+1, 1, vocab_size)
    layer_list = torch.stack(layer_list) # (num_labels+1, len, vocab_size)
    combination_vector = combination_vector.to(layer_list.device)
    temp_logits = combination_vector * layer_list  # (num_labels+1, len, vocab_size)
    final_logits = temp_logits.sum(dim=0)  # (len, vocab_size)

    return final_logits

def create_vector(args):
    if args.wo_final:
        num_labels = len(args.concerned_labels.split(','))
        tensor = torch.randn((num_labels, 32000))
    else:
        # combine final
        num_labels = len(args.concerned_labels.split(','))
        tensor = torch.randn((num_labels+1, 32000))

    # num_labels = len(args.concerned_labels.split(','))
    # tensor = torch.zeros((num_labels+1, 32000))
    # tensor[-1, :] = 1

    # num_labels = len(args.concerned_labels.split(','))
    # tensor = torch.randn((num_labels, 32000))

    vector = torch.nn.Parameter(tensor.cuda())
    return vector

def main(args):
    # load llm model
    ModelClass = getattr(Model, args.model.replace('-', '_'))
    LlmModel = ModelClass(args)

    TrainLoader = dataloader(args, LlmModel.tokenizer, 'train')
    ValidLoader = dataloader(args, LlmModel.tokenizer, 'valid')

    # initialize vector
    combination_vector = create_vector(args)
    
    # set optimizer, scheduler and loss function
    total_steps = len(TrainLoader) * args.epoch / args.batch_size
    optimizer = torch.optim.Adam([combination_vector], lr=args.lr)  

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    # train
    model = LlmModel.model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print(args)
    print('start training...')
    # track average loss
    step = 0
    batch_loss = 0
    total_loss = 0.0

    for epoch in range(args.epoch):
        pbar = tqdm(TrainLoader, desc=f"Epoch: {epoch+1}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            outputs = model(
                        input_ids=batch['input_ids'],
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True
                    )
            # import pdb; pdb.set_trace()
            final_logits = combine(args, model, outputs.hidden_states, batch['question_ids'], combination_vector)
            final_logits = final_logits.to(batch['label_ids'].device)
            single_loss = loss_fn(final_logits, batch['label_ids'])

            batch_loss += single_loss

            step += 1

            if step and not step % (args.batch_size):
                # Although the batch size of the dataloader is 1
                # the parameter feedback is always after the actual batch size
                # where step refers to the number of steps in the dataloader
                batch_loss_avg = batch_loss / args.batch_size
                batch_loss_avg.backward()
                optimizer.step()
                scheduler.step()

                # updata total loss
                total_loss += batch_loss_avg.item()
                batch_loss = 0
            
            if step % (args.print_every * args.batch_size) == 0:
                avg_loss = total_loss / args.print_every
                pbar.set_description(f"Epoch: {epoch+1}, Step: {step//args.batch_size}, Avg Loss: {avg_loss:.4f}")
                total_loss = 0.0

            # save & eval
            if step % (args.save_every * args.batch_size) == 0:
                save_and_valid(args, step//args.batch_size, combination_vector, model, ValidLoader)


def parse_args():
    parser = argparse.ArgumentParser(description='train')
    
    # model config
    parser.add_argument('--model', default='', type=str, help='llama1-7b')
    parser.add_argument('--data-path', default='', type=str, help='training data and validing data')

    # train config
    parser.add_argument('--epoch', default=3, type=int, help='training epochs')
    parser.add_argument('--batch-size', default=16, type=int, help='training batch size')
    parser.add_argument('--max-len', default=512, type=int, help='max length of sentence')
    parser.add_argument('--save-path', default='', type=str, help='save path of finetuned models')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--concerned-labels', default='0,2,4,6,8,10,12,14,16', type=str)
    parser.add_argument('--wo-final', default=False, action="store_true")

    # log config
    parser.add_argument('--print-every', default=100, type=int, help='print log and valid model every few steps')
    parser.add_argument('--save-every', default=100, type=int, help='save model every few steps')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)