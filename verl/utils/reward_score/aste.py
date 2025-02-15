import re
import random
import ast
import operator
import sys


def extract_solution(solution_str):
    """Extract the solution from the generated string.
    Returning another string not a python list.
    """
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def prepare_prediction(elements: list):
    """Prepare the prediction for evaluation.
    Need to add more corner cases here.
    """
    # Convert the prediction to a list of tuples
    if len(elements) == 1:
        modified = [' ; '.join(elements[0])]
    else:
        mod_tup = []
        for tup in elements:
            at, ot, sp = tup
            at = at or 'NULL'
            ot = ot or 'NULL'
            sp = sp or 'NULL'
            mod_tup.append((at, ot, sp))
        modified = [' ; '.join(r) for r in mod_tup]
    return modified


def compute_f1_score_single(pred_pt: list, gold_pt: list):
    """
    Function to compute F1 scores with pred and gold pairs/triplets
    for a single example, while considering aspect, opinion, and sentiments
    all together. The entire tuple needs to be correct for a match.
    There is no reward for a partial match.
    Ref: https://github.com/IsakZhang/Generative-ABSA/blob/main/eval_utils.py
    """
    # number of true postive, gold standard, predicted aspect terms
    n_tp, n_gold, n_pred = 0, 0, 0

    n_gold += len(gold_pt)
    n_pred += len(pred_pt)

    for t in pred_pt:
        candidates = [e.lower().strip() for e in gold_pt]  # modifying for LLM generated response
        # candidates = gold_pt  # original
        if t.lower() in candidates:
            n_tp += 1

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for the ASTE task.

    Args:
        solution_str: the model generated solution text
        ground_truth: dictionary containing target tuples
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer

        Example: {'target': "['shipping ; free ; POS ', ' price ; best ; POS ', ' budget ; fits ; POS']"}
    """
    # print(f"Solution: {solution_str}")
    # print(f"Ground truth: {ground_truth}")

    target_str = ground_truth['target']
    target = eval(target_str, {"__builtins__": None}, {})
    target = [t.strip() for t in target]
    solution = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 32) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target}")
        print(f"\nGenerated string: {solution_str}")
        print(f"\nExtracted solution: {solution}")
        print(f"________________________________")

    if solution is None:
        if do_print:
            print(f"No solution found between <answer> and </answer>")
        return 0

    # there is some string between <answer> and </answer>
    try:
        solution_list = eval(solution)  # should be a list of strings
        # solution_list = prepare_prediction(solution_list)
        scores = compute_f1_score_single(pred_pt=solution_list, gold_pt=target)
        part_score = scores['f1']
        if do_print:
            if part_score > 0.1:
                print(f"PARTIAL MATCHED: predicted: {solution_list} = target: {target}")
            else:
                print(f"WRONG: predicted: {solution_list}, target: {target}, F1-score: {part_score}")
        return part_score
    except Exception as e:
        if do_print:
            print(f"Could not evaluate the solution string: {solution}")
            print(f"Error: {e}")
        return format_score
