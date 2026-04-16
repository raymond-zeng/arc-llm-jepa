import argparse
import json
import signal
import sys
import math
import wandb
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")


def parse_code(generated_text):
    code_str = generated_text.strip()
    if "<solution>" in code_str:
        code_str = code_str.split("<solution>", 1)[1]
        if "</solution>" in code_str:
            code_str = code_str.split("</solution>", 1)[0]
        code_str = code_str.strip()

    # Also handle markdown blocks
    if "```python" in code_str:
        code_str = code_str.split("```python", 1)[1]
        if "```" in code_str:
            code_str = code_str.split("```", 1)[0]
        code_str = code_str.strip()
    elif "```" in code_str:
        code_str = code_str.split("```", 1)[1]
        if "```" in code_str:
            code_str = code_str.split("```", 1)[0]
        code_str = code_str.strip()
    return code_str


def evaluate_solution(code_str, test_examples, timeout=5):
    runnable_code = code_str

    local_vars = {}
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        exec(runnable_code, globals(), local_vars)
        signal.alarm(0)
    except TimeoutError:
        signal.alarm(0)
        return False, f"Timeout during code execution (exceeded {timeout}s)"
    except Exception as e:
        signal.alarm(0)
        return False, f"Compilation/Execution error:\n{e}"

    if 'solve' not in local_vars:
        return False, "The generated code did not define a 'solve' function."

    solve_func = local_vars['solve']

    for test_idx, test_case in enumerate(test_examples):
        test_input = [list(row) for row in test_case['input']]
        expected_output = tuple(tuple(row) for row in test_case['output'])
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            predicted_output = solve_func(test_input)
            # Normalize to tuple of tuples for comparison
            if isinstance(predicted_output, (list, tuple)):
                predicted_output = tuple(tuple(row) if isinstance(row, (list, tuple)) else row for row in predicted_output)
            signal.alarm(0)
            if predicted_output != expected_output:
                return False, f"Output mismatch on Test Case {test_idx + 1}"
        except TimeoutError:
            signal.alarm(0)
            return False, f"Timeout on Test Case {test_idx + 1} (exceeded {timeout}s)"
        except Exception as e:
            signal.alarm(0)
            return False, f"Exception during execution on Test Case {test_idx + 1}: {e}"

    return True, "All test cases passed."


def main():
    parser = argparse.ArgumentParser(description="Evaluate finetuned LLM on ARC-AGI")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Base model name/path")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter checkpoint (uses HF generate)")
    parser.add_argument("--merged_model", type=str, default=None, help="Path to merged model (uses vLLM)")
    parser.add_argument("--eval_set", type=str, default="evaluation",
                        choices=["training", "evaluation"],
                        help="Which ARC challenge set to evaluate on")
    parser.add_argument("--num_examples", type=int, default=400, help="Number of examples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum generated tokens")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for HF generate (ignored with vLLM)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--prompt_max_length", type=int, default=8192, help="Maximum prompt length for HF generation")
    parser.add_argument("--wandb_project", type=str, default="arc-eval-finetuned", help="Wandb project name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--max_model_len", type=int, default=None, help="Model context length for vLLM")

    args = parser.parse_args()

    if not args.adapter_path and not args.merged_model:
        parser.error("Must provide either --adapter_path or --merged_model")

    use_vllm = args.merged_model is not None

    if not args.no_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    # Load model
    if use_vllm:
        from vllm import LLM, SamplingParams
        print(f"Loading merged model {args.merged_model} with vLLM...")
        tokenizer = AutoTokenizer.from_pretrained(args.merged_model)
        llm = LLM(model=args.merged_model, dtype="bfloat16",
                   tokenizer=args.merged_model,
                   max_model_len=args.max_model_len if args.max_model_len else None)
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            stop=["</solution>", "```\n"],
        )
    else:
        print(f"Loading base model {args.base_model}...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Add the same special tokens that were added during finetuning
        special_tokens = ["<|predictor_1|>", "<|predictor_2|>", "<|predictor_3|>", "<|predictor_4|>", "<|predictor_5|>",
                          "<|predictor_6|>", "<|predictor_7|>", "<|predictor_8|>", "<|predictor_9|>", "<|predictor_10|>",
                          "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|perception|>"]
        new_tokens = [token for token in special_tokens if token not in tokenizer.vocab]
        if new_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
            print(f"Added {len(new_tokens)} special tokens to match finetuning")

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, device_map="auto", torch_dtype=torch.bfloat16
        )
        if new_tokens:
            model.resize_token_embeddings(len(tokenizer))

        print(f"Loading LoRA adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model.eval()

    # Load evaluation challenges
    challenge_file = f"arc-prize-2024/arc-agi_{args.eval_set}_challenges.json"
    solution_file = f"arc-prize-2024/arc-agi_{args.eval_set}_solutions.json"

    print(f"Loading challenges from {challenge_file}...")
    with open(challenge_file, 'r') as f:
        challenges = json.load(f)

    has_solutions = False
    solutions = {}
    try:
        with open(solution_file, 'r') as f:
            solutions = json.load(f)
        has_solutions = True
        print(f"Loaded solutions from {solution_file}")
    except FileNotFoundError:
        print(f"No solution file found at {solution_file}, using challenge test cases")

    task_ids = list(challenges.keys())
    to_evaluate = min(args.num_examples, len(task_ids))
    task_ids = task_ids[:to_evaluate]

    system_msg = (
        "Write python code to solve the following puzzles. Use only standard Python "
        "(no external libraries). Define a function `solve(grid)` that takes a 2D "
        "list of integers and returns a 2D list of integers. Return only the complete "
        "function inside <solution>...</solution> tags."
    )

    if use_vllm:
        # === vLLM path: build all prompts, generate at once ===
        prompts = []
        for task_id in task_ids:
            task_data = challenges[task_id]
            prompt_messages = [
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": (
                        f"Train examples:\n{json.dumps(task_data['train'])}\n\n"
                        "Respond with the complete solve function wrapped in "
                        "<solution>...</solution>."
                    ),
                }
            ]
            if tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = (
                    f"{system_msg}\n\n"
                    f"Train examples:\n{json.dumps(task_data['train'])}\n\n"
                    "Respond with the complete solve function wrapped in "
                    "<solution>...</solution>.\n\n"
                )
            prompts.append(prompt)

        print(f"Generating solutions for {len(prompts)} puzzles...")
        outputs = llm.generate(prompts, sampling_params)

        correct_puzzles = 0
        total_puzzles = 0

        for i, output in enumerate(tqdm(outputs, desc="Evaluating")):
            generated_text = output.outputs[0].text
            code_str = parse_code(generated_text)

            task_id = task_ids[i]
            task_data = challenges[task_id]

            if has_solutions and task_id in solutions:
                test_examples = [
                    {"input": task_data['test'][k]['input'], "output": solutions[task_id][k]}
                    for k in range(len(solutions[task_id]))
                ]
            else:
                test_examples = task_data['test']

            success, msg = evaluate_solution(code_str, test_examples)
            total_puzzles += 1
            if success:
                correct_puzzles += 1

            if not args.no_wandb:
                wandb.log({
                    "current_puzzle": total_puzzles,
                    "puzzles_solved": correct_puzzles,
                    "accuracy": correct_puzzles / total_puzzles
                })

            if i % 20 == 0:
                print(f"\nExample {total_puzzles} - {task_id}")
                print("--- Generated Code Snippet ---")
                print(code_str[:300] + "...")
                print(f"Result: {'✅ Pass' if success else '❌ Fail'} ({msg})")

    else:
        # === HuggingFace path: batched generation ===
        correct_puzzles = 0
        total_puzzles = 0
        batches = math.ceil(to_evaluate / args.batch_size)
        truncated_prompts = 0

        for b in tqdm(range(batches), desc="Processing batches"):
            batch_task_ids = task_ids[b * args.batch_size: (b + 1) * args.batch_size]
            limit = len(batch_task_ids)

            prompts = []
            prompt_lengths = []
            for j, task_id in enumerate(batch_task_ids):
                task_data = challenges[task_id]
                prompt_messages = [
                    {"role": "system", "content": system_msg},
                    {
                        "role": "user",
                        "content": (
                            f"Train examples:\n{json.dumps(task_data['train'])}\n\n"
                            "Respond with the complete solve function wrapped in "
                            "<solution>...</solution>."
                        ),
                    }
                ]
                prompt = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt)
                prompt_lengths.append(len(tokenizer(prompt, add_special_tokens=False)["input_ids"]))

            truncated_prompts += sum(length > args.prompt_max_length for length in prompt_lengths)

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.prompt_max_length,
                add_special_tokens=False,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            input_length = inputs["input_ids"].shape[-1]

            for j in range(limit):
                generated_ids = outputs[j][input_length:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                code_str = parse_code(generated_text)

                task_id = batch_task_ids[j]
                task_data = challenges[task_id]

                if has_solutions and task_id in solutions:
                    test_examples = [
                        {"input": task_data['test'][k]['input'], "output": solutions[task_id][k]}
                        for k in range(len(solutions[task_id]))
                    ]
                else:
                    test_examples = task_data['test']

                success, msg = evaluate_solution(code_str, test_examples)
                total_puzzles += 1
                if success:
                    correct_puzzles += 1

                if not args.no_wandb:
                    wandb.log({
                        "batch": b,
                        "current_puzzle": total_puzzles,
                        "puzzles_solved": correct_puzzles,
                        "accuracy": correct_puzzles / total_puzzles
                    })

                if j == 0 and b % 5 == 0:
                    print(f"\nExample {total_puzzles} - {task_id}")
                    print("--- Generated Code Snippet ---")
                    print(code_str[:300] + "...")
                    print(f"Result: {'✅ Pass' if success else '❌ Fail'} ({msg})")

        if truncated_prompts:
            print(f"Warning: {truncated_prompts} prompts exceeded prompt_max_length={args.prompt_max_length} and were truncated during evaluation.")

    accuracy = correct_puzzles / total_puzzles
    print(f"\nFinal Accuracy: {correct_puzzles}/{total_puzzles} ({accuracy * 100:.2f}%)")
    if not args.no_wandb:
        wandb.log({"final_accuracy": accuracy, "total_solved": correct_puzzles, "total_evaluated": total_puzzles})
        wandb.finish()


if __name__ == "__main__":
    main()
