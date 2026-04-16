import json
import sys

def build_dataset():
    # Load distilled solutions
    with open('datasets/distilled_solutions.json', 'r') as f:
        solutions = json.load(f)

    # Load challenges
    with open('arc-prize-2024/arc-agi_training_challenges.json', 'r') as f:
        challenges = json.load(f)

    # Output file
    out_path = 'datasets/arc_training.jsonl'
    count = 0

    with open(out_path, 'w') as out_f:
        for task_id, task_data in challenges.items():
            if task_id not in solutions:
                continue

            code = solutions[task_id]

            messages = [
                {"role": "system", "content": "Write python code to solve the following puzzles. Use only standard Python (no external libraries). Define a function `solve(grid)` that takes a 2D list of integers and returns a 2D list of integers. Only write code and nothing else."},
                {"role": "user", "content": json.dumps(task_data['train'])},
                {"role": "assistant", "content": code}
            ]

            out_f.write(json.dumps({"messages": messages}) + "\n")
            count += 1

    print(f"Dataset generated at {out_path} ({count} examples)")

    # Validate JSONL format
    with open(out_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error on line {i+1}: {e}")
                sys.exit(1)

    print(f"Dataset validation passed. {i+1} valid JSON objects found in JSONL file.")

if __name__ == "__main__":
    build_dataset()
