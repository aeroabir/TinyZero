"""
Preprocess dataset for simlin2 task - solve a system of two linear equations with integer coefficients
"""

import re
import os
import pandas as pd
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple, Dict
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse

def read_dataset(
    num_samples: int,
    seed_value: int = 42,
    file_name: str = None
) -> Dict:
    """
        Read GRPO dataset compatibility prediction and do some processing
    Args:
        num_samples: Number of samples to generate
        seed_value: Random seed for reproducibility
        file_name: original RL data
    Returns:
        List of tuples containing (prompt, response, ground_truth)
    """

    seed(seed_value)
    all_data = pd.read_json(file_name, lines=True)
    all_data = all_data[all_data.num_items <= 5]
    if len(all_data) < num_samples:
        num_samples = len(all_data)
    all_data = all_data.sample(n=num_samples)
    df = {'prompt': [], 'response': [], 'gt': []}
    for _, row in all_data.iterrows():
        df['prompt'].append(row['prompt'][0]['content'])
        df['response'].append(row['completion'][0]['content'])
        df['gt'].append(row['gt'])

    return df


def make_prefix(dp, template_type):
    prompt = dp['prompt']
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: {prompt} Show your logic in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> compatible </answer>.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/azureuser/localfiles/data/polyvore_cp')
    parser.add_argument('--train_file', default='compatibility_text_based_grpo_train_14272.jsonl')
    parser.add_argument('--dev_file', default='compatibility_text_based_grpo_dev_1352.jsonl')
    parser.add_argument('--test_file', default='compatibility_text_based_grpo_test_73213.jsonl')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=10000)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'polyvore'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size
    train_file = os.path.join(args.local_dir, args.train_file)
    dev_file = os.path.join(args.local_dir, args.dev_file)

    train_dataset = read_dataset(num_samples=TRAIN_SIZE, seed_value=100, file_name=train_file)
    dev_dataset = read_dataset(num_samples=TEST_SIZE, seed_value=200, file_name=dev_file)

    train_dataset = Dataset.from_dict(train_dataset)
    dev_dataset = Dataset.from_dict(dev_dataset)

    assert len(train_dataset['prompt']) >= TRAIN_SIZE
    assert len(dev_dataset['prompt']) >= TEST_SIZE

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type) # to add some syntax
            # question = example['prompt']
            solution = {
                "target": example['gt']
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    dev_dataset = dev_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    dev_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
