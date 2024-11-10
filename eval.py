import argparse
import json
import os

from datasets import load_dataset
from src.eval_utils import compare_answer_with_groundtruth
from src.utils import Query, generate_response
from tqdm import tqdm
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--results_path", type=str, default="./output/eval_data/")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--dataset_name", type=str, default="gsm8k")
    parser.add_argument("--device_num", type=int, default=8)
    parser.add_argument("--eval_data_file", type=str, default="./data/gsm8k_original.json")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--mode", type=str, default="cot")
    return parser.parse_args()


def eval_dataset(dataset, args):
    model = LLM(model=args.model_name_or_path, tensor_parallel_size=args.device_num, seed=42)
    sampling_params = SamplingParams(max_tokens=args.max_length, temperature=args.temperature, seed=42)

    os.makedirs(args.results_path, exist_ok=True)

    queries = []
    for item in dataset:
        question = item["question"]
        # PoT mode: add a prompt to the question
        if args.mode == "pot":
            question += " Let's write a program."

        new_query = Query(
            item["item_id"],
            question,
            args.dataset_name,
            item["answer_cot"] if "answer_cot" in item else None,
            item["answer_value"],
            args.mode,
        )
        queries.append(new_query)

    generate_response(queries, model, sampling_params)

    # calculate the number of correct answers
    correct_counts = 0
    for query in queries:
        if compare_answer_with_groundtruth(query.pred_value[0], query.target_value):
            correct_counts += 1

    eval_accuracy = correct_counts / len(queries)
    print(f"Correct answers: {correct_counts}")
    print(f"Eval accuracy: {eval_accuracy * 100:.3f}%")

    results_path = os.path.join(args.results_path, f"{args.dataset_name}_res.json")
    with open(results_path, "w") as f:
        json.dump([query.to_dict() for query in queries], f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dataset = load_dataset("json", data_files=args.eval_data_file)["train"]
    eval_dataset(dataset, args)
