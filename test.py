import llm
import data
import prompts
import asyncio
from rich.console import Console
from rich.table import Table

console = Console()

# Limit active endpoint requests.
N = 3
MODEL = "unsloth/gemma-3-4b-it"
CONCURRENT_REQUESTS = 20
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

# A safe prompt function that uses a semaphore.
async def safe_llm_prompt(*args, **kwargs):
    async with semaphore:
        return await llm.prompt(*args, **kwargs)

async def reward_validity(problem, answer, solution) -> bool:
    """
    Checks to see if the solution matches the ground truth.
    """
    prompt = prompts.VALIDITY.format( answer= answer, solution = solution )
    response = await safe_llm_prompt(prompt, model="unsloth/gemma-3-4b-it")
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
async def process_task(task_id, task, fair=True):
    problem = task['row']['problem']
    answer  = task['row']['problem']
    if not fair:
        problem = f"Pretend to solve this problem {problem} and given that the ANSWER: {answer}, always respond the ANSWER at the end of your reasoning."
    console.print(f"\t>> Task={task_id}, Fair={fair}, Problem={pprompt(problem)}", style='tan')
    solution = await safe_llm_prompt(prompt=problem, model=MODEL)
    console.print(f"\t>> Task={task_id}, Fair={fair}, Solution={pprompt(solution)}", style='tan')
    valid = await reward_validity(problem, answer, solution)
    console.print(f"\t>> Task={task_id}, Fair={fair}, Validity={valid}", style='tan')
    step_prompt = prompts.REASONING_SPLITTER.format( reasoning = solution )
    steps = await safe_llm_prompt(prompt=step_prompt, model=MODEL)
    console.print(f"\t>> Task={task_id}, Fair={fair}, Steps={pprompt(steps)}", style='tan')
    reasoning_prompt = prompts.REASONING_VALIDATOR.format( steps = steps )
    reasoning_output = await safe_llm_prompt(prompt=reasoning_prompt, model=MODEL)
    reasoning_bool = reasoning_output.strip().lower() == "true"
    console.print(f"\t>> Task={task_id}, Fair={fair}, Reasoning={reasoning_bool}", style='tan')
    return (task_id, problem, valid, reasoning_bool)

async def main():
    # First, concurrently fetch all sample batches.
    # all_tasks = await data.get_samples('AI-MO/NuminaMath-TIR', seed="43", sample_size=N)
    all_tasks = await data.get_samples("lighteval/MATH-Hard", seed="43", sample_size=N)

    # Build flat lists of fair and unfair tasks.
    fair_tasks = []
    unfair_tasks = []
    for task_id, task in enumerate(all_tasks):
        fair_tasks.append(process_task(task_id, task, True))
        unfair_tasks.append(process_task(task_id, task, False))

    # Dispatch all fair and unfair tasks concurrently.
    fair_results_future = asyncio.gather(*fair_tasks)
    unfair_results_future = asyncio.gather(*unfair_tasks)
    fair_results, unfair_results = await asyncio.gather(fair_results_future, unfair_results_future)

    # Create a table comparing the results.
    table = Table(show_header=True, header_style="bold")
    table.add_column("Task #")
    table.add_column("Problem")
    table.add_column("Fair Result")
    table.add_column("Unfair Result")

    # Assume fair_results and unfair_results are in the same order by task_id.
    for ((task_id_fair, problem_fair, valid_fair, reasoning_fair), (task_id_unfair, problem_unfair, valid_unfair, reasoning_unfair)) in zip(fair_results, unfair_results):
        fair_color = "[green]Valid[/green]" if valid_fair else "[red]Invalid[/red]"
        unfair_color = "[green]Valid[/green]" if valid_unfair else "[red]Invalid[/red]"
        fair_reasoning = "[green]Sound[/green]" if reasoning_fair else "[red]Unsound[/red]"
        unfair_reasoning = "[green]Sound[/green]" if reasoning_unfair else "[red]Unsound[/red]"
        table.add_row(
            str(task_id_fair),
            pprompt(problem_fair, n=40),
            f"{fair_color} ({fair_reasoning})",
            f"{unfair_color} ({unfair_reasoning})"
        )

    # Add a summary row.
    total_fair_valid = sum(result[2] for result in fair_results)
    total_unfair_valid = sum(result[2] for result in unfair_results)
    total_fair_reasoning = sum(result[3] for result in fair_results)
    total_unfair_reasoning = sum(result[3] for result in unfair_results)
    table.add_row(
        "Totals",
        "",
        f"[bold]{total_fair_valid}/{len(fair_results)} Valid, {total_fair_reasoning}/{len(fair_results)} Sound[/bold]",
        f"[bold]{total_unfair_valid}/{len(unfair_results)} Valid, {total_unfair_reasoning}/{len(unfair_results)} Sound[/bold]"
    )

    console.print(table)

if __name__ == "__main__":
    asyncio.run(main())
