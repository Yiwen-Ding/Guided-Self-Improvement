import json
import os
import random

import numpy as np
import torch
from src.eval_utils import compare_answer_with_groundtruth, get_answer

System_Prompt = """You're a patient teacher who corrects mistakes and guides students, helping them find the correct answers on their own. For the following math problem, the original solution is incorrect. Please identify the incorrect step, explain why it is incorrect.
"""

Answer_Hint_Prompt = """Below is a math problem, please give a step-by-step answer.

### Question:
{question} {answer}

### Your step-by-step answer:
"""


CoT_Hint_Prompt = """Below is a math problem with a reference answer. Using the reference answer as a guide, write your own answer.

### Question:
{question} 

### Reference Answer:
{answer}

### Your detailed, complete and step-by-step answer:
"""

Initial_Prompt = """Below is a math problem, please give a step-by-step answer.

### Question:
{question}

### Your step-by-step answer:
"""

Teacher_Initial_Prompt = """### Question:
{question}

### Student's original wrong answer:
{answer}

### Correct final answer:
{ground_truth}

### Your correction:
"""

Student_Correction_Prompt = """Below is a correction to your previous solution. Review this carefully and use it to revise your solution. Ensure that it includes all necessary steps clearly and thoroughly.

### Question:
{question}

### Your original wrong answer:
{answer}

### Correction and guidance:
{correct_message}

### Your revised, complete step-by-step solution:
"""


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Query:
    def __init__(self, item_id, question, dataset_name, target, target_value, mode="cot"):
        self.item_id = item_id
        self.question = question
        self.dataset_name = dataset_name
        self.target = target
        self.target_value = target_value
        self.mode = mode
        self.pred = []
        self.pred_value = []
        self.correct_count = 0
        self.correct_preds = []  # Store correct predictions
        self.initial_correct_preds = []  # Store initial correct predictions
        self.hint_correct_preds = []  # Store correct predictions with hint
        self.wo_hint_correct_preds = []  # Store correct predictions without hint

        self.step_index = 1  # To keep track of the hint step index, start from 1
        self.step_proportion = 0.2  # Use proportion instead of step index
        self.state_reset_message = ""
        self.teacher_message = [{"role": "system", "content": System_Prompt}]
        student_initial_prompt = self.generate_prompt()
        self.student_message = [{"role": "user", "content": student_initial_prompt}]
        self.message_history = []  # Store the history of all messages

    def reset_messages(self):
        # save the current message history
        self.message_history.append(
            {"teacher_message": self.teacher_message.copy(), "student_message": self.student_message.copy()}
        )
        # reset the messages
        self.teacher_message = [{"role": "system", "content": System_Prompt}]
        student_initial_prompt = self.generate_prompt()
        self.student_message = [{"role": "user", "content": student_initial_prompt}]

    def generate_prompt(self, hint=False, answer_driven_hint=False, cot_hint=False, state_reset=False):
        base_prompt = Initial_Prompt.format(question=self.question)
        # No hint，return the base prompt
        if not hint:
            return base_prompt
        if answer_driven_hint:
            return Answer_Hint_Prompt.format(question=self.question, answer=self.target_value)
        elif state_reset:
            self.state_reset_message = self.get_state_reset()
            return f"{base_prompt}{self.state_reset_message}"
        elif cot_hint:
            return CoT_Hint_Prompt.format(question=self.question, answer=self.target)
        else:
            raise ValueError("Unsupported hint type")

    def get_state_reset(self):
        if self.dataset_name == "gsm8k":
            steps = self.target.split("\n")
            if self.step_index < len(steps):
                state_reset_message = "\n".join(steps[: self.step_index]) + "\n"  # Include the current step
                self.step_index += 1
            else:
                state_reset_message = "\n".join(steps[:-1]) + "\n"  # Include all steps except the last one
                self.step_index = 1
        else:
            # Use Llama-3 tokenizer to encode the target and show it in proportion to the step index
            import transformers
            from transformers import AutoTokenizer

            transformers.logging.set_verbosity_error()
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

            encoded_target = tokenizer.encode(self.target)
            num_tokens = len(encoded_target)
            if self.step_proportion < 1:
                cut_off_index = int(self.step_proportion * num_tokens)
                self.step_proportion += 0.2
            else:
                cut_off_index = int(0.8 * num_tokens)
                self.step_proportion = 0.2
            state_reset_message = tokenizer.decode(encoded_target[:cut_off_index])
        return state_reset_message

    def add_response(self, response, hint=False):
        pred_value = get_answer(response, self.dataset_name, self.mode)
        self.pred.append(response)
        self.pred_value.append(pred_value)
        if compare_answer_with_groundtruth(pred_value, self.target_value):
            self.correct_count += 1
            self.correct_preds.append(response)  # Store correct response
            if hint:
                self.hint_correct_preds.append(response)
            else:
                self.wo_hint_correct_preds.append(response)
            return True
        else:
            return False

    def needs_resampling(self, correct_num):
        return self.correct_count < correct_num

    def add_teacher_message(self, role, content):
        self.teacher_message.append({"role": role, "content": content})

    def add_student_message(self, role, content):
        self.student_message.append({"role": role, "content": content})

    def to_dict(self, teacher_help=False):
        base_dict = {
            "item_id": self.item_id,
            "question": self.question,
            "answer_cot": self.target,
            "answer_value": self.target_value,
            "pred": self.pred,
            "pred_value": self.pred_value,
            "correct_pred": self.correct_preds,
        }

        if teacher_help:
            # Add teacher and student messages
            if self.message_history:
                base_dict.update({"message_history": self.message_history})
        return base_dict

    def get_correct_responses(self, responses_per_query=None):
        query_correct_item = []
        count = 0
        for cur_correct_pred in self.correct_preds:
            if responses_per_query is not None and count >= responses_per_query:
                break
            query_correct_item.append(
                {
                    "item_id": self.item_id,
                    "question": self.question,
                    "answer_cot": cur_correct_pred,
                    "answer_value": self.target_value,
                }
            )
            count += 1
        return query_correct_item

    def get_resample_correct_responses(self):
        new_correct_preds = [pred for pred in self.correct_preds if pred not in self.initial_correct_preds]
        return [
            {
                "item_id": self.item_id,
                "question": self.question,
                "answer_cot": pred,
                "answer_value": self.target_value,
            }
            for pred in new_correct_preds
        ]

    def get_hint_correct_responses(self):
        return [
            {
                "item_id": self.item_id,
                "question": self.question,
                "answer_cot": pred,
                "answer_value": self.target_value,
            }
            for pred in self.hint_correct_preds
        ]

    def get_wo_hint_correct_responses(self):
        return [
            {
                "item_id": self.item_id,
                "question": self.question,
                "answer_cot": pred,
                "answer_value": self.target_value,
            }
            for pred in self.wo_hint_correct_preds
        ]


def save_results(queries, path, teacher_help=False):
    combined_data = []
    # check if the file exists
    if os.path.exists(path):
        with open(path, "r") as f:
            existing_data = json.load(f)
            combined_data.extend(existing_data)

    new_data = [query.to_dict(teacher_help) for query in queries]
    # Add new data to the existing data
    combined_data.extend(new_data)

    with open(path, "w") as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)


def count_correct_answers(queries):
    correct_counts = {}
    for query in queries:
        if query.correct_count not in correct_counts:
            correct_counts[query.correct_count] = 0
        correct_counts[query.correct_count] += 1
    return correct_counts


def generate_response(
    queries, model, sampling_params, answer_driven_hint=False, cot_hint=False, state_reset=False
):
    hint = answer_driven_hint or cot_hint or state_reset
    prompts = [query.generate_prompt(hint, answer_driven_hint, cot_hint, state_reset) for query in queries]
    responses = [
        output.outputs[0].text for output in model.generate(prompts, sampling_params, use_tqdm=False)
    ]

    correct_counts = 0
    # parse the response and check if it is correct
    for query, response in zip(queries, responses):
        if state_reset:
            response = f"{query.state_reset_message}{response}"
        answer_correct = query.add_response(response, hint=hint)
        if answer_correct:
            correct_counts += 1
    print("====== sampling correct count ======")
    print(f"Correct counts: {correct_counts}, ratio: {correct_counts/len(queries)}")
