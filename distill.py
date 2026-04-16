"""Distill ARC solutions from a large model via OpenRouter.

For each training task, prompts the model to write a plain Python `solve` function,
validates it against all training examples, and saves correct solutions.
"""

import argparse
import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
from dotenv import load_dotenv

import requests
from tqdm import tqdm

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are an expert at solving ARC-AGI puzzles. "
    "Given training examples (input/output grid pairs), write a Python function `solve(grid)` "
    "that takes a 2D list of integers and returns a 2D list of integers. "
    "Use only standard Python and numpy (no external libraries). "
    "Provide the function definition inside <solution>...</solution> tags."
)


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")


def parse_response(text):
    """Extract reasoning and code from XML-tagged response."""
    if not text:
        return "", ""
    reasoning = ""
    if "<reasoning>" in text and "</reasoning>" in text:
        reasoning = text.split("<reasoning>")[1].split("</reasoning>")[0].strip()

    code = ""
    if "<solution>" in text and "</solution>" in text:
        code = text.split("<solution>")[1].split("</solution>")[0].strip()
        # Strip markdown fences if model wraps code inside solution tags
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

    return reasoning, code


def validate_solution(code_str, examples, timeout=5):
    """Run the solve function against all examples. Returns (success, message)."""
    local_vars = {}
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        exec(code_str, {}, local_vars)
        signal.alarm(0)
    except TimeoutError:
        signal.alarm(0)
        return False, "Timeout during compilation"
    except Exception as e:
        signal.alarm(0)
        return False, f"Compilation error: {e}"

    if "solve" not in local_vars:
        return False, "No 'solve' function defined"

    solve_func = local_vars["solve"]

    for i, ex in enumerate(examples):
        test_input = [list(row) for row in ex["input"]]
        expected = [list(row) for row in ex["output"]]
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            result = solve_func(test_input)
            signal.alarm(0)
        except TimeoutError:
            signal.alarm(0)
            return False, f"Timeout on example {i}"
        except Exception as e:
            signal.alarm(0)
            return False, f"Runtime error on example {i}: {e}"

        # Normalize to list-of-lists for comparison
        if isinstance(result, (list, tuple)):
            result = [list(row) if isinstance(row, (list, tuple)) else row for row in result]

        if result != expected:
            return False, f"Output mismatch on example {i}"

    return True, "All examples passed"


def query_model(model, task_data, api_key, temperature=0.2, max_tokens=16384, reasoning_budget=8192):
    """Query OpenRouter for a solution."""
    user_content = json.dumps(task_data["train"], indent=None)

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_budget:
        body["reasoning"] = {"max_tokens": reasoning_budget}

    for retry in range(5):
        resp = requests.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=180,
        )
        if resp.status_code == 429:
            wait = 2 ** retry
            print(f"Rate limited, retrying in {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"OpenRouter error: {data['error']}")
        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning") or ""
        return reasoning, content

    resp.raise_for_status()  # will raise on the last 429


def main():
    parser = argparse.ArgumentParser(description="Distill ARC solutions from a large model")
    parser.add_argument("--model", type=str, default="anthropic/claude-sonnet-4",
                        help="OpenRouter model identifier")
    parser.add_argument("--output", type=str, default="datasets/distilled_solutions.json",
                        help="Output path for validated solutions")
    parser.add_argument("--traces_output", type=str, default="datasets/distilled_traces.json",
                        help="Output path for reasoning traces")
    parser.add_argument("--max_attempts", type=int, default=1,
                        help="Max generation attempts per task")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=12288)
    parser.add_argument("--reasoning_budget", type=int, default=8192,
                        help="Max tokens for model reasoning (0 to disable)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel API workers")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    parser.add_argument("--debug", action="store_true",
                        help="Run on 1 puzzle, print full LLM response, then exit")

    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: Set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    # Load challenges and test solutions
    with open("arc-prize-2024/arc-agi_training_challenges.json") as f:
        challenges = json.load(f)
    with open("arc-prize-2024/arc-agi_training_solutions.json") as f:
        test_solutions = json.load(f)

    # Load existing progress if resuming
    solutions = {}
    traces = {}
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            solutions = json.load(f)
        if Path(args.traces_output).exists():
            with open(args.traces_output) as f:
                traces = json.load(f)
        print(f"Resuming with {len(solutions)} existing solutions")

    task_ids = [tid for tid in challenges if tid not in solutions]
    print(f"Tasks to process: {len(task_ids)} ({len(solutions)} already done)")

    if args.debug:
        task_id = task_ids[0]
        task_data = challenges[task_id]
        print(f"\n=== DEBUG: task {task_id} ===")
        api_reasoning, content = query_model(args.model, task_data, api_key,
                          temperature=args.temperature,
                          max_tokens=args.max_tokens,
                          reasoning_budget=args.reasoning_budget)
        print(f"\n=== API Reasoning Field ===\n{api_reasoning[:2000] if api_reasoning else '(empty)'}\n")
        print(f"\n=== Content Field ===\n{content}\n")
        reasoning, code = parse_response(content)
        # Use API reasoning field if XML reasoning is empty
        if not reasoning and api_reasoning:
            reasoning = api_reasoning
        print(f"=== Parsed Reasoning ===\n{reasoning}\n")
        print(f"=== Parsed Code ===\n{code}\n")
        if code:
            ok, msg = validate_solution(code, task_data["train"])
            print(f"=== Validation: {'PASS' if ok else 'FAIL'} — {msg} ===")
        else:
            print("=== No code extracted ===")
        return

    def fetch_solution(task_id, attempt):
        """Worker: call API and parse response. Validation happens on main thread."""
        task_data = challenges[task_id]
        temp = args.temperature + attempt * 0.2
        try:
            api_reasoning, content = query_model(args.model, task_data, api_key,
                              temperature=min(temp, 1.0),
                              max_tokens=args.max_tokens,
                              reasoning_budget=args.reasoning_budget)
            reasoning, code = parse_response(content)
            if not reasoning and api_reasoning:
                reasoning = api_reasoning
            return task_id, attempt, reasoning, code, None
        except requests.exceptions.RequestException as e:
            return task_id, attempt, None, None, f"API error: {e}"
        except Exception as e:
            return task_id, attempt, None, None, f"Error: {e}"

    def save_progress():
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(solutions, f, indent=2)
        with open(args.traces_output, "w") as f:
            json.dump(traces, f, indent=2)

    solved = 0
    failed = 0
    # Track which tasks still need solving and their current attempt number
    pending = {tid: 0 for tid in task_ids}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit initial batch
        futures = {}
        for task_id in task_ids:
            fut = executor.submit(fetch_solution, task_id, 0)
            futures[fut] = task_id

        pbar = tqdm(total=len(task_ids), desc="Distilling")

        for fut in as_completed(futures):
            task_id = futures[fut]
            _, attempt, reasoning, code, error = fut.result()

            if error:
                tqdm.write(f"{task_id} attempt {attempt+1}: {error}")

            # Validate on main thread (signal.alarm requires main thread)
            ok = False
            if code:
                ok, msg = validate_solution(code, challenges[task_id]["train"])
                # Also validate against test examples to catch hardcoded solutions
                if ok and task_id in test_solutions:
                    test_examples = [
                        {"input": t["input"], "output": out}
                        for t, out in zip(challenges[task_id]["test"], test_solutions[task_id])
                    ]
                    ok, msg = validate_solution(code, test_examples)
                    if not ok:
                        tqdm.write(f"{task_id} attempt {attempt+1}: passed train but failed test ({msg})")

            if ok:
                solutions[task_id] = code
                if reasoning:
                    traces[task_id] = reasoning
                solved += 1
                del pending[task_id]
                pbar.update(1)
            else:
                next_attempt = attempt + 1
                if next_attempt < args.max_attempts:
                    pending[task_id] = next_attempt
                    new_fut = executor.submit(fetch_solution, task_id, next_attempt)
                    futures[new_fut] = task_id
                else:
                    failed += 1
                    del pending[task_id]
                    pbar.update(1)

            if (solved + failed) % 20 == 0 and (solved + failed) > 0:
                save_progress()

        pbar.close()

    save_progress()

    total = len(task_ids)
    print(f"\nDone: {solved}/{total} solved, {failed}/{total} failed")
    print(f"Total solutions: {len(solutions)}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
