"""
Preprocess dataset for simlin2 task - solve a system of two linear equations with integer coefficients
"""

import re
import os
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple, Dict
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse


def gen_dataset_cubic(
    num_samples: int,
    min_number: int = 1,
    max_number: int = 100,
    seed_value: int = 42,
) -> Dict:
    """
        Generate dataset for cubic equations
    Args:
        num_samples: Number of samples to generate
        min_number: Minimum value for provided numbers
        max_number: Maximum value for provided numbers
        seed_value: Random seed for reproducibility
    Returns:
        List of tuples containing (coefficients, solution)
    """
    count = 0
    seed(seed_value)
    df = {'A': [], 'b': [], 'x': []}
    min_x_number, max_x_number = -min_number, max_number
    min_y_number, max_y_number = -min_number, max_number
    min_z_number, max_z_number = -min_number, max_number
    while count < num_samples:
        r1 = randint(min_x_number, max_x_number)
        r2 = randint(min_y_number, max_y_number)
        r3 = randint(min_z_number, max_z_number)
        c0 = 1
        c1 = -(r1 + r2 + r3)
        c2 = (r1 * r2 + r1 * r3 + r2 * r3)
        c3 = -r1 * r2 * r3
        df['A'].append(f"[{c0}, {c1}, {c2}, {c3}]")
        df['b'].append('not used')  # for compatibility
        df['x'].append(f"[{r1:.0f}, {r2:.0f}, {r3:.0f}]")
        count += 1
    return df


def gen_dataset(
    num_samples: int,
    min_number: int = 1,
    max_number: int = 100,
    seed_value: int = 42,
    real_soln: bool = False,
    tolerance: float = 1e-06
) -> Dict:
    """
        Generate dataset for simultaneous linear equation solution task
    Args:
        num_samples: Number of samples to generate
        min_number: Minimum value for provided numbers
        max_number: Maximum value for provided numbers
        seed_value: Random seed for reproducibility
        real_soln: Boolean flag to indicate whether the solutions should be integer or not
    Returns:
        List of tuples containing (coefficients, rhs, solution)
    """

    count = 0
    seed(seed_value)
    samples = []
    df = {'A': [], 'b': [], 'x': []}
    min_x_number, max_x_number = -min_number, max_number
    min_y_number, max_y_number = -min_number, max_number
    while count < num_samples:
        a11 = randint(min_number, max_number)
        a12 = randint(min_number, max_number)
        a21 = randint(min_number, max_number)
        # delta = randint(min_number, max_number)
        # a22 = int((delta + a21*a12)/a11)
        a22 = randint(min_number, max_number)
        if a11*a22 - a12*a21 != 0:
            count += 1
            if not real_soln:
                # first decide the solution, then get the rhs
                x = randint(min_x_number, max_x_number)
                y = randint(min_y_number, max_y_number)
                b1 = a11 * x + a12 * y
                b2 = a21 * x + a22 * y
            else:
                # solution for any rhs
                b1 = randint(-min_number, max_number)
                b2 = randint(-min_number, max_number)
                determinant = a11 * a22 - a12 * a21
                x = (1./determinant) * (a22 * b1 - a12 * b2)
                y = (1./determinant) * (-a21 * b1 + a11 * b2)
                x = round(x, 2)
                y = round(y, 2)
                # adjust the rhs
                # b1 = a11 * x + a12 * y
                # b2 = a21 * x + a22 * y

            samples.append((a11, a12, a21, a22, b1, b2, x, y))

    # Check the samples for correctness
    for sample in tqdm(samples, total=len(samples)):
        a11, a12, a21, a22, b1, b2, x0, y0 = sample
        determinant = a11 * a22 - a12 * a21
        assert determinant != 0, 'Determinant is equal to 0!!'
        x = (1./determinant) * (a22 * b1 - a12 * b2)
        y = (1./determinant) * (-a21 * b1 + a11 * b2)
        eq1 = a11 * x + a12 * y - b1
        eq2 = a21 * x + a22 * y - b2
        assert abs(eq1) < tolerance, f'Equation-1: {eq1}'
        assert abs(eq2) < tolerance, f'Equation-2: {eq2}'
        assert abs(x-x0) < tolerance, f'X: {x} != {x0}'
        assert abs(y-y0) < tolerance, f'Y: {y} != {y0}'
        df['A'].append(f"[{a11}, {a12}, {a21}, {a22}]")
        df['b'].append(f"[{b1}, {b2}]")
        df['x'].append(f"[{x:.2f}, {y:.2f}]")
    return df


def make_prefix(dp, template_type):
    coeff = dp['A']
    rhs = dp['b']
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: A system of two linear equations with coefficients {coeff} (arranged as [a11, a12, a21, a22]) and the right hand side {rhs} (arranged as [b1, b2]) is given. Solve the system of equations. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> [1, 2] </answer>.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix


def make_prefix_cubic(dp, template_type):
    coeff = dp['A']
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between a User and an Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: A cubic equation with coefficients {coeff} (arranged as [c0, c1, c2, c3], so that the equation is c0*x^3 + c1*x^2 + c2*x + c3 =0) is given. Solve the equation to find all the three roots. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> [1, 2, 3] </answer>.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/countdown')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--num_operands', type=int, default=6)
    parser.add_argument('--max_target', type=int, default=1000)
    parser.add_argument('--min_number', type=int, default=1)
    parser.add_argument('--max_number', type=int, default=100)
    parser.add_argument('--train_size', type=int, default=327680)
    parser.add_argument('--test_size', type=int, default=1024)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--complexity', type=str, default='integer')

    args = parser.parse_args()

    data_source = 'simlin2'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    if args.complexity == "integer":
        # Make one type of dataset
        train_dataset = gen_dataset(num_samples=TRAIN_SIZE, seed_value=100, real_soln=False)
        test_dataset = gen_dataset(num_samples=TEST_SIZE, seed_value=200, real_soln=False)

    elif args.complexity == "float":
        train_dataset = gen_dataset(num_samples=TRAIN_SIZE, seed_value=100, real_soln=True)
        test_dataset = gen_dataset(num_samples=TEST_SIZE, seed_value=200, real_soln=True)

    elif args.complexity == "mixed":
        # Make a combination of difficulties
        train_dataset = gen_dataset(num_samples=TRAIN_SIZE//3,
                                    seed_value=100,
                                    real_soln=True,
                                    max_number=30,
                                    tolerance=1e-01)
        test_dataset = gen_dataset(num_samples=TEST_SIZE//3,
                                   seed_value=200,
                                   real_soln=True,
                                   max_number=30,
                                   tolerance=1e-01)
        # integer roots
        train_dataset_2 = gen_dataset(num_samples=TRAIN_SIZE//3,
                                      seed_value=100,
                                      max_number=30,
                                      real_soln=False)
        test_dataset_2 = gen_dataset(num_samples=TEST_SIZE//3,
                                     seed_value=200,
                                     max_number=30,
                                     real_soln=False)
        # cubic polynomial
        train_dataset_3 = gen_dataset_cubic(num_samples=TRAIN_SIZE//3+2,
                                            min_number=10,
                                            max_number=30,
                                            seed_value=100)
        test_dataset_3 = gen_dataset_cubic(num_samples=TEST_SIZE//3+2,
                                           min_number=10,
                                           max_number=30,
                                           seed_value=200)
        for k in train_dataset:
            train_dataset[k] += train_dataset_2[k]
            train_dataset[k] += train_dataset_3[k]
        for k in test_dataset:
            test_dataset[k] += test_dataset_2[k]
            test_dataset[k] += test_dataset_3[k]

    train_dataset = Dataset.from_dict(train_dataset)
    test_dataset = Dataset.from_dict(test_dataset)

    assert len(train_dataset['x']) >= TRAIN_SIZE, f"{len(train_dataset['x'])}, {TRAIN_SIZE}"
    assert len(test_dataset['x']) >= TEST_SIZE, f"{len(test_dataset['x'])}, {TEST_SIZE}"

    def make_map_fn(split):
        def process_fn(example, idx):
            if example['b'] == 'not used':
                question = make_prefix_cubic(example, template_type=args.template_type)
                solution = {
                    "target": example['x'],
                    "numbers": f"A: {example['A']}"
                }
                qtype = 'cubic'
            else:
                question = make_prefix(example, template_type=args.template_type)
                solution = {
                    "target": example['x'],
                    "numbers": f"A: {example['A']}, b: {example['b']}"
                }
                qtype = 'linear'
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
                    'type': qtype,
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
