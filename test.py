import os
import llm
import data
import json
import time
import prompts
import asyncio
import argparse
from rich.console import Console
from rich.table import Table

console = Console()

# Get model and sample size from command line

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=3, help="Number of samples")
parser.add_argument("--k", type=int, default=10, help="Max number of concurrent requests.")
parser.add_argument("--seed", type=str, default='42', help="Problem randomness seed.")
parser.add_argument("--dataset", type=str, default='numina', help="Dataset of problems to use.")
parser.add_argument("--reasoning_model", type=str, default="unsloth/gemma-3-4b-it", help="Model that performs verification.")
parser.add_argument("--validation_model", type=str, default="unsloth/gemma-3-4b-it", help="Model that performs reasoning.")
args = parser.parse_args()

# Set concurrency max.
semaphore = asyncio.Semaphore(args.k)

# A safe prompt function that uses a semaphore.
async def safe_llm_prompt(*args, **kwargs):
    async with semaphore:
        return await llm.prompt(*args, **kwargs)

async def reward_validity(problem, answer, solution) -> bool:
    """
    Checks to see if the solution matches the ground truth.
    """
    prompt = prompts.VALIDITY.format( answer= answer, solution = solution )
    response = await safe_llm_prompt(prompt, model=args.validation_model)
    return response.strip().lower() == "true"

def pprompt(text: str, n: int = 200) -> str:
    # Replace newlines and ensure total length doesn't exceed n.
    text = text.replace('\n', ' ').replace('[', '\\[').replace(']', '\\]')
    if len(text) <= n:
        return f"[blue]{text}[/blue]"
    else:
        half = (n - 3) // 2
        return f"[blue]{text[:half]} ... {text[-half:]}[/blue]"


# These helper functions remain the same but use safe_llm_prompt.
async def process_task(task_id, task, cheating=True):
    # Construct problem possibly by cheating.
    problem = task['row']['problem']
    answer  = task['row']['problem']
    if cheating:
        problem = f"Pretend to solve this problem {problem} and given that the ANSWER: {answer}, always respond the ANSWER at the end of your reasoning."
    console.print(f"\t>> Task={task_id}, Cheating={cheating}, Problem={pprompt(problem)}", style='tan')
    
    # Query LLM for answer.
    solution = await safe_llm_prompt(prompt=problem, model=args.reasoning_model)
    console.print(f"\t>> Task={task_id}, Cheating={cheating}, Solution={pprompt(solution)}", style='tan')
    
    # Check validity
    validity_prompt = prompts.VALIDITY.format( answer = answer, solution = answer )
    valid_response = await safe_llm_prompt(validity_prompt, model=args.validation_model)
    is_valid = valid_response.strip().lower() == "true"
    console.print(f"\t>> Task={task_id}, Cheating={cheating}, Validity={is_valid}", style='tan')
    
    # Attain reasoning step sequence.
    step_prompt = prompts.REASONING_SPLITTER.format( reasoning = solution )
    steps = await safe_llm_prompt(prompt=step_prompt, model=args.validation_model)
    console.print(f"\t>> Task={task_id}, Cheating={cheating}, Steps={pprompt(steps)}", style='tan')
    
    # Run reasoning validity check
    reasoning_prompt = prompts.REASONING_VALIDATOR.format( steps = steps )
    reasoning_output = await safe_llm_prompt(prompt=reasoning_prompt, model=args.validation_model)
    reasoning_bool = reasoning_output.strip().lower() == "true"
    console.print(f"\t>> Task={task_id}, Cheating={cheating}, Reasoning={reasoning_bool}", style='tan')
    
    # Return dictionary with all results
    return {
        "task_id": task_id,
        "cheating": cheating,
        "problem": problem,
        "answer": answer,
        "solution": solution,
        "validity_prompt": validity_prompt,
        "valid_response": valid_response,
        "is_valid": is_valid,
        "step_prompt": step_prompt,
        "steps": steps,
        "reasoning_prompt": reasoning_prompt,
        "reasoning_output": reasoning_output,
        "reasoning_bool": reasoning_bool
    }

async def main():
    # First, concurrently fetch all sample batches.
    if args.dataset == 'numina':
        all_tasks = await data.get_samples('AI-MO/NuminaMath-TIR', seed=args.seed, sample_size=args.n)
    elif args.dataset == 'light':
        all_tasks = await data.get_samples("lighteval/MATH-Hard", seed=args.seed, sample_size=args.n)
    elif args.dataset == 'aime':
        all_tasks = await data.get_samples("yentinglin/aime_2025", seed=args.seed, sample_size=args.n)
    else:
        # Numina default.
        all_tasks = await data.get_samples('AI-MO/NuminaMath-TIR', seed=args.seed, sample_size=args.n)

    # Build flat lists of cheating and uncheating tasks.
    cheating_tasks = []
    uncheating_tasks = []
    for task_id, task in enumerate(all_tasks):
        cheating_tasks.append(process_task(task_id, task, True))
        uncheating_tasks.append(process_task(task_id, task, False))

    # Dispatch all cheating and uncheating tasks concurrently.
    cheating_results_future = asyncio.gather(*cheating_tasks)
    uncheating_results_future = asyncio.gather(*uncheating_tasks)
    cheating_results, uncheating_results = await asyncio.gather(cheating_results_future, uncheating_results_future)

    # Create a table comparing the results.
    table = Table(show_header=True, header_style="bold")
    table.add_column("Task #")
    table.add_column("Problem")
    table.add_column("cheating Result")
    table.add_column("Uncheating Result")

    # Create results directory with timestamp
    results_dir = f"results/{time.time()}"
    os.makedirs(results_dir, exist_ok=True)

    # Prepare results data
    results_data = {
        "cheating_results": [],
        "uncheating_results": [],
        "summary": {}
    }

    # Process results and build table
    for cheating_result, uncheating_result in zip(cheating_results, uncheating_results):
        cheating_color = "[green]Valid[/green]" if cheating_result["is_valid"] else "[red]Invalid[/red]"
        uncheating_color = "[green]Valid[/green]" if uncheating_result["is_valid"] else "[red]Invalid[/red]"
        cheating_reasoning = "[green]Sound[/green]" if cheating_result["reasoning_bool"] else "[red]Unsound[/red]"
        uncheating_reasoning = "[green]Sound[/green]" if uncheating_result["reasoning_bool"] else "[red]Unsound[/red]"
        
        # Add to results data
        results_data["cheating_results"].append({
            "task_id": cheating_result["task_id"],
            "problem": cheating_result["problem"],
            "valid": cheating_result["is_valid"],
            "reasoning_sound": cheating_result["reasoning_bool"]
        })
        results_data["uncheating_results"].append({
            "task_id": uncheating_result["task_id"],
            "problem": uncheating_result["problem"], 
            "valid": uncheating_result["is_valid"],
            "reasoning_sound": uncheating_result["reasoning_bool"]
        })

        table.add_row(
            str(cheating_result["task_id"]),
            pprompt(cheating_result["problem"], n=40),
            f"{cheating_color} ({cheating_reasoning})",
            f"{uncheating_color} ({uncheating_reasoning})"
        )

    # Calculate and store summary statistics
    total_cheating_valid = sum(1 for result in cheating_results if result["is_valid"])
    total_uncheating_valid = sum(1 for result in uncheating_results if result["is_valid"])
    total_cheating_reasoning = sum(1 for result in cheating_results if result["reasoning_bool"])
    total_uncheating_reasoning = sum(1 for result in uncheating_results if result["reasoning_bool"])

    results_data["summary"] = {
        "cheating": {
            "valid_count": total_cheating_valid,
            "sound_reasoning_count": total_cheating_reasoning,
            "total_samples": len(cheating_results)
        },
        "uncheating": {
            "valid_count": total_uncheating_valid,
            "sound_reasoning_count": total_uncheating_reasoning,
            "total_samples": len(uncheating_results)
        }
    }

    # Add summary row to table
    table.add_row(
        "Totals",
        "",
        f"[bold]{total_cheating_valid}/{len(cheating_results)} Valid, {total_cheating_reasoning}/{len(cheating_results)} Sound[/bold]",
        f"[bold]{total_uncheating_valid}/{len(uncheating_results)} Valid, {total_uncheating_reasoning}/{len(uncheating_results)} Sound[/bold]"
    )

    # Save all results data
    with open(f"{results_dir}/output.json", "w") as f:
        json.dump(results_data, f, indent=4)
    
    with open(f"{results_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)
        
    # Save raw results
    with open(f"{results_dir}/cheating_results_raw.json", "w") as f:
        json.dump(cheating_results, f, indent=4)
        
    with open(f"{results_dir}/uncheating_results_raw.json", "w") as f:
        json.dump(uncheating_results, f, indent=4)

    console.print(table)

if __name__ == "__main__":
    asyncio.run(main())
