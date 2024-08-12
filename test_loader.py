# 文件名：dataset_loader.py
import os
import json
import pandas as pd
import ast
import re


def Gsm8kLoader(args):
    print("Loading Gsm8k...")

    list_data_dict = []

    def clean(answer):
        ANS_RE = re.compile(r"The answer is (\-?[0-9\.\,]+)")
        match = ANS_RE.search(answer)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(".", "")
            return match_str
        else:
            return "[invalid]"
    
    df = pd.read_csv(os.path.join(args.data_path, 'test/test.csv'), header=None)
    for i in range(1, df.shape[0]): # skip first row
        question, answer = df.iloc[i]

        new_item = dict(
            question=question,
            answer=clean(answer))
        item = new_item
        list_data_dict.append(item)

    return list_data_dict

def StraQALoader(args):
    print("Loading StraQA...")

    list_data_dict = []

    def clean(answer):
        ANS_RE = re.compile(r"So the answer is (\w+)")
        match = ANS_RE.search(answer)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(".", "")
            return match_str == 'yes'
        else:
            raise ValueError('Answer is wrong. Please check the test data.')
    
    df = pd.read_csv(os.path.join(args.data_path, 'test/test.csv'), header=None)
    for i in range(1, df.shape[0]): # skip first row
        question, answer = df.iloc[i]

        new_item = dict(
            question=question,
            answer=clean(answer))
        item = new_item
        list_data_dict.append(item)

    return list_data_dict

def WikiTextLoader(args):
    print("Loading WikiText...")

    list_data_dict = []

    df = pd.read_csv(os.path.join(args.data_path, 'test/test.csv'), header=None)
    for i in range(1, df.shape[0]): # skip first row
        question, answer = df.iloc[i]

        new_item = dict(
            question=question,
            answer=answer)
        item = new_item
        list_data_dict.append(item)

    return list_data_dict

def WikiNewsLoader(args):
    print("Loading WikiNews...")

    list_data_dict = []

    df = pd.read_csv(os.path.join(args.data_path, 'test/test.csv'), header=None)
    for i in range(1, df.shape[0]): # skip first row
        question, answer = df.iloc[i]

        new_item = dict(
            question=question,
            answer=answer)
        item = new_item
        list_data_dict.append(item)

    return list_data_dict

def BookCorpusLoader(args):
    print("Loading BookCorpus...")

    list_data_dict = []

    df = pd.read_csv(os.path.join(args.data_path, 'test/test.csv'), header=None)
    for i in range(1, df.shape[0]): # skip first row
        question, answer = df.iloc[i]

        new_item = dict(
            question=question,
            answer=answer)
        item = new_item
        list_data_dict.append(item)

    return list_data_dict
