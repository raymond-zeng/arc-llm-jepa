import argparse
import json
import signal
import sys
import wandb
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


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
    parser = argparse.ArgumentParser(description="Evaluate LLM on ARC-AGI")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model name/path")
    parser.add_argument("--num_examples", type=int, default=400, help="Number of examples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--wandb_project", type=str, default="arc-eval", help="Wandb project name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    print(f"Loading model {args.model_name} with vLLM...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm = LLM(model=args.model_name, dtype="bfloat16")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stop=["```\n"],
    )

    # Load evaluation challenges and solutions
    with open('arc-prize-2024/arc-agi_evaluation_challenges.json', 'r') as f:
        challenges = json.load(f)

    with open('arc-prize-2024/arc-agi_evaluation_solutions.json', 'r') as f:
        solutions = json.load(f)
        
    task_ids = list(challenges.keys())
    to_evaluate = min(args.num_examples, len(task_ids))
    task_ids = task_ids[:to_evaluate]

    # Build all prompts
    system_msg = (
        "Write python code to solve the following puzzles. Use only standard Python "
        "(no external libraries). Define a function `solve(grid)` that takes a 2D "
        "list of integers and returns a 2D list of integers. Only write code and nothing else."
    )
    
    prompts = []
    for task_id in task_ids:
        task_data = challenges[task_id]
        prompt_messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": (
                    f"Train examples:\n{json.dumps(task_data['train'])}\n\n"
                    "Only write code and nothing else."
                ),
            }
        ]
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback for models without a chat template (e.g. CodeGemma)
            prompt = (
                f"{system_msg}\n\n"
                f"Train examples:\n{json.dumps(task_data['train'])}\n\n"
                "Only write code and nothing else.\n\n"
            )
        prompts.append(prompt)

    # Generate all at once — vLLM handles batching internally
    print(f"Generating solutions for {len(prompts)} puzzles...")
    outputs = llm.generate(prompts, sampling_params)

    correct_puzzles = 0
    total_puzzles = 0

    for i, output in enumerate(tqdm(outputs, desc="Evaluating")):
        generated_text = output.outputs[0].text
        code_str = parse_code(generated_text)

        task_id = task_ids[i]
        task_data = challenges[task_id]
        test_examples = [
            {"input": task_data['test'][k]['input'], "output": solutions[task_id][k]}
            for k in range(len(solutions[task_id]))
        ]

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
            print(code_str[:200] + "...")
            print(f"Result: {'✅ Pass' if success else '❌ Fail'} ({msg})")
                
    accuracy = correct_puzzles / total_puzzles
    print(f"\nFinal Accuracy: {correct_puzzles}/{total_puzzles} ({accuracy*100:.2f}%)")
    if not args.no_wandb:
        wandb.log({"final_accuracy": accuracy, "total_solved": correct_puzzles, "total_evaluated": total_puzzles})
        wandb.finish()

if __name__ == "__main__":
    main()
