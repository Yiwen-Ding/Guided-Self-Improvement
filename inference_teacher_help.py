import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from openai import OpenAI
from src.utils import (
    Query,
    Student_Correction_Prompt,
    Teacher_Initial_Prompt,
    count_correct_answers,
    generate_response,
    save_results,
)
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MAX_RETRIES = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--teacher_model", type=str, default="llama3-70b-instruct")
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


def create_chat_response_by_messages(
    model="llama3-70b-instruct",
    client=None,
    messages=None,
    max_tokens=1024,
    temperature=1.0,
    top_p=0.95,
):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return response.choices[0].message.content

        except (TypeError, ValueError) as e:
            print(f"Error occurred: {e}. Retrying {retries + 1}/{MAX_RETRIES}...")
            retries += 1

    print("Max retries reached. Returning None.")
    return None


def resample_teacher_guide(
    resample_queries, student_model, teacher_model, teacher_client, tokenizer, sampling_params, args
):
    teacher_responses = {}

    def process_query(query):
        teacher_message = Teacher_Initial_Prompt.format(
            question=query.question, answer=query.pred[-1], ground_truth=query.target
        )

        query.add_teacher_message("user", teacher_message)

        # Teacher model generates response
        cur_response = create_chat_response_by_messages(
            model=teacher_model,
            client=teacher_client,
            messages=query.teacher_message,
            max_tokens=1024,
            temperature=1.0,
            top_p=0.95,
        )
        return query.item_id, cur_response

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=48) as executor:
        future_to_query = {executor.submit(process_query, query): query for query in resample_queries}

        for future in tqdm(
            as_completed(future_to_query), total=len(resample_queries), desc="Teacher Guide Progress"
        ):
            item_id, response = future.result()
            teacher_responses[item_id] = response

    correct_prompts = []
    # Append teacher responses to corresponding query messages
    for query in resample_queries:
        teacher_response = teacher_responses.get(query.item_id)
        query.add_teacher_message("assistant", teacher_response)
        correct_prompt = Student_Correction_Prompt.format(
            question=query.question, answer=query.pred[-1], correct_message=teacher_response
        )
        correct_prompts.append(correct_prompt)
        query.add_student_message("user", correct_prompt)

    student_responses = [
        output.outputs[0].text
        for output in student_model.generate(correct_prompts, sampling_params, use_tqdm=False)
    ]

    # Append student responses to corresponding query messages
    for query, student_response in zip(resample_queries, student_responses):
        query.add_response(student_response)
        query.reset_messages()
        query.add_student_message("assistant", student_response)


def inference_dataset(dataset, args):
    student_model = LLM(model=args.model_name_or_path, tensor_parallel_size=args.device_num)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    sampling_params = SamplingParams(max_tokens=args.max_length, temperature=args.temperature)

    # Set API base to use vLLM's API server.
    openai_api_key = ""
    openai_api_base = ""

    teacher_client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

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

    # Step 1: Initial inference
    for cur_sample_num in tqdm(range(args.sample_num)):
        generate_response(queries, student_model, sampling_params)

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

        resample_teacher_guide(
            resample_queries,
            student_model,
            args.teacher_model,
            teacher_client,
            tokenizer,
            sampling_params,
            args,
        )

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

    save_results(queries, os.path.join(args.results_path, "results.json"), teacher_help=True)


def sample_train_dataset(args):
    dataset = load_dataset("json", data_files=args.inference_data_file)["train"]
    inference_dataset(dataset, args)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sample_train_dataset(args)
