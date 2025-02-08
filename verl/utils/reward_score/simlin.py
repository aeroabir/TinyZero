import re
import random
import ast
import operator


def extract_equation(numbers_str):
    """Extract the coefficients of the linear equations
        It will appear like
        "numbers": f"A: {example['A']}, b: {example['b']}"
        Example: 'numbers': 'A: [6, 27, 95, 442], b: [4, 19]'
    """
    coeff_pattern = r'\[[\d,\.\s\-]+\]'
    try:
        match = re.findall(coeff_pattern, numbers_str)
        matches = list(match)
        coeffs = eval(matches[0], {"__builtins__": None}, {})
        rhs = eval(matches[1], {"__builtins__": None}, {})
        return coeffs, rhs
    except Exception as e:
        print(e)
        print(f"Input: {numbers_str}")
        match = re.findall(coeff_pattern, numbers_str)
        matches = list(match)
        print(matches)

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
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


def validate_solution(solution):
    """Validate that the solution has only two numbers."""
    return len(solution) == 2


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, square bracket, comma, and whitespace
        allowed_pattern = r'^\[[\d,\.\s\-+]+\]'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for simlin task.

    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer

        "target": example['x'],
        "numbers": f"A: {example['A']}, b: {example['b']}"
        Example: 'numbers': 'A: [6, 27, 95, 442], b: [4, 19]',
                 'target': '[14.425, -3.057]'
    """
    THRESHOLD = 1e-1  # 1e-5
    target_str = ground_truth['target']
    numbers_str = ground_truth['numbers']

    # extract the equation coefficients and rhs
    coeffs, rhs = extract_equation(numbers_str)
    target = eval(target_str, {"__builtins__": None}, {})
    solution = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target_str} | Numbers: {numbers_str}")
        print(f"Extracted solution: {solution}")
        print(f"Solution string: {solution_str}")

    if solution is None:
        if do_print:
            print(f"No solution found")
        return 0

    # Validate that the solution contains exactly two numbers
    # if not validate_solution(solution):
    #     if do_print:
    #         print(f"Invalid solution")
    #     return format_score

    # Evaluate equation
    try:
        soln = evaluate_equation(solution)
        if soln is None:
            if do_print:
                print(f"Could not evaluate the solution")
            return format_score

        # Multiply the prediction by the coefficient matrix
        result = [coeffs[0] * soln[0] + coeffs[1] * soln[1], coeffs[2] * soln[0] + coeffs[3] * soln[1]]
        if abs(result[0] - rhs[0]) < THRESHOLD and abs(result[1] - rhs[1]) < THRESHOLD:  # Account for floating point precision
            if do_print:
                print(f"Predicted solution: {solution} = {target}")
            return score
        else:
            if do_print:
                print(f"Wrong result: predicted = {solution}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score