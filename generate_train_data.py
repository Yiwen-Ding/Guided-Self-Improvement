import argparse
import json
import random
import re

from src.eval_utils import compare_answer_with_groundtruth


def parse_args():
    parser = argparse.ArgumentParser(description="Generate training data")
    parser.add_argument("--original_data_path", type=str, required=True, help="Path to the original data")
    parser.add_argument(
        "--pre_iter_data_path", type=str, required=True, help="Path to the previous iteration data"
    )
    parser.add_argument(
        "--cur_sampled_data_path", type=str, required=True, help="Path to the current sampled data"
    )
    parser.add_argument(
        "--next_iter_data_path", type=str, required=True, help="Path to the next iteration data"
    )
    parser.add_argument("--include_original_data", action="store_true", help="Include original data")
    parser.add_argument(
        "--include_pre_iter_data", action="store_true", help="Include previous iteration data"
    )
    parser.add_argument("--check_result", action="store_true", help="Check if the equation is correct")

    return parser.parse_args()


def check_equation(equation):
    if equation.find("=") == -1:
        return False
    lhs = equation.split("=")[0]
    rhs = equation.split("=")[-1]
    try:
        lhs_result = str(eval(str(lhs)))
        if compare_answer_with_groundtruth(lhs_result, rhs):
            return True
        else:
            return False
    except BaseException:
        return False


def check_reasoning_paths(solution):
    pattern = r"<<([^>]*)>>"  # Match everything inside << >>
    matches = re.findall(pattern, solution)  # Find all matches
    equation_flag = True
    for match in matches:
        if not check_equation(match):
            equation_flag = False
    return equation_flag


def main(args):
    # read original data
    if args.include_original_data:
        with open(args.original_data_path, "r") as file:
            original_data = json.load(file)
    # read previous iteration data
    elif args.include_pre_iter_data:
        with open(args.pre_iter_data_path, "r") as file:
            pre_iter_data = json.load(file)

    # read current sampled data
    with open(args.cur_sampled_data_path, "r") as file:
        data = json.load(file)

    incorrect_count = 0

    new_data = []
    for item in data:
        item_id = item["item_id"]
        question = item["question"]
        target_value = item["answer_value"]
        for correct_pred in item["correct_pred"]:
            if args.check_result and not check_reasoning_paths(correct_pred):
                incorrect_count += 1
                continue
            new_item = {
                "item_id": item_id,
                "question": question,
                "answer_cot": correct_pred,
                "answer_value": target_value,
            }
            new_data.append(new_item)

    if args.check_result:
        print(f"incorrect equations: {incorrect_count}")

    print(f"Number of correctly sampled data: {len(new_data)}")

    # merge data
    if args.include_original_data:
        new_data.extend(original_data)
    elif args.include_pre_iter_data:
        new_data.extend(pre_iter_data)

    print(f"Number of merged data: {len(new_data)}")

    # write the current iteration data
    with open(args.next_iter_data_path, "w") as file:
        json.dump(new_data, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
