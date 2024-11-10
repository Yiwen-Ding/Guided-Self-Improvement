import argparse
import json
import os

from datasets import load_dataset
from src.utils import Query, count_correct_answers, generate_response, save_results
from tqdm import tqdm
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--results_path", type=str, default="./output/inference_data/")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--dataset_name", type=str, default="gsm8k")
    parser.add_argument("--min_correct_num", type=int, default=3)
    parser.add_argument("--device_num", type=int, default=8)
    parser.add_argument("--sample_num", type=int, default=10)
    parser.add_argument("--max_resample_num", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--mode", type=str, default="cot")
    parser.add_argument("--inference_data_file", type=str, default="./data/gsm8k_original.json")
    return parser.parse_args()


def inference_dataset(dataset, args):
    model = LLM(model=args.model_name_or_path, tensor_parallel_size=args.device_num)
    sampling_params = SamplingParams(max_tokens=args.max_length, temperature=args.temperature)

    os.makedirs(args.results_path, exist_ok=True)

    queries = [
        Query(
            item["item_id"],
            item["question"],
            args.dataset_name,
            item["answer_cot"],
            item["answer_value"],
            args.mode,
        )
        for item in dataset
    ]

    # Initial sampling with hint=False
    for cur_sample_num in tqdm(range(args.sample_num)):
        generate_response(queries, model, sampling_params)

        cur_correct_count = sum(1 for query in queries if query.correct_count > 0)
        cur_distribution = count_correct_answers(queries)
        print(f"====== Coverage: After initial {cur_sample_num} samples with hint=False ======")
        print(f"Pass@1 query: {cur_correct_count}, ratio {cur_correct_count/len(queries) * 100:.3f}%")
        print(f"Initial correct counts distribution: {cur_distribution}\n")

    # Record initial correct reasoning paths
    for query in queries:
        query.initial_correct_preds = query.correct_preds.copy()

    # Save results before resampling
    save_results(queries, os.path.join(args.results_path, "results_before_resample.json"))

    # tqdm progress bar
    progress_bar = tqdm(total=args.max_resample_num, desc="Resampling Progress")

    # Loop for resampling if needed
    total_sample_count = 0
    while total_sample_count < args.max_resample_num:
        resample_queries = [query for query in queries if query.needs_resampling(args.min_correct_num)]
        # If queries are empty, all queries have reached the target
        if len(resample_queries) == 0:
            print("All queries have reached the required correct count.")
            break

        generate_response(resample_queries, model, sampling_params, cot_hint=True)

        total_sample_count += 1
        progress_bar.update(1)

        # Stat correct_query and correct_counts after each sampling
        correct_query = sum(1 for query in queries if query.correct_count > 0)
        correct_counts = count_correct_answers(queries)

        print(f"====== Coverage: After {total_sample_count} resamples  ======")
        print(f"Pass@1 query: {correct_query}, ratio {correct_query/len(queries) * 100:.3f}%")
        print(f"Correct counts distribution: {correct_counts}\n")

    # Save new correct responses after resampling
    recorrect_response = []
    for query in queries:
        recorrect_response.extend(query.get_resample_correct_responses())

    results_file_path = os.path.join(args.results_path, "resample_results.json")

    if os.path.exists(results_file_path):
        # If the file exists, merge
        with open(results_file_path, "r") as f:
            existing_data = json.load(f)
        recorrect_response.extend(existing_data)

    with open(results_file_path, "w") as f:
        json.dump(recorrect_response, f, indent=4, ensure_ascii=False)

    # Close the progress bar
    progress_bar.close()

    save_results(queries, os.path.join(args.results_path, "results.json"))


def sample_train_dataset(args):
    dataset = load_dataset("json", data_files=args.inference_data_file)["train"]
    inference_dataset(dataset, args)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sample_train_dataset(args)
