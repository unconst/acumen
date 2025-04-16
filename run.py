# The MIT License (MIT)
# © 2025 Acumen

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import re
import llm
import sys
import data
import time
import comms
import torch
import asyncio
import datetime
import argparse
import traceback
import bittensor as bt
from rich.console import Console
from async_lru import alru_cache
from typing import List, Dict, Tuple, Optional, Union

def grpo_scores(rewards: torch.Tensor) -> torch.Tensor:
    """
    Computes the pareto GRPO score for each point in a reward set using Pareto dominance count.
    Uses PyTorch operations for computation.
    """
    n = rewards.size(0)
    scores = torch.ones(n, dtype=torch.float32)    
    # Create comparison matrices
    rewards_i = rewards.unsqueeze(0).expand(n, n)  # Shape: [n, n]
    rewards_j = rewards.unsqueeze(1).expand(n, n)  # Shape: [n, n]
    # Calculate dominance for all pairs
    dominates = torch.logical_and(
        (rewards_i >= rewards_j),  # Remove .all(dim=-1) since rewards is 1D
        (rewards_i > rewards_j)    # Remove .any(dim=-1) since rewards is 1D
    ).float()
    # Sum up dominance scores
    scores = scores + dominates.sum(dim=0) # Change dim=1 to dim=0 since rewards is 1D
    return scores

async def check_format( solution: Dict ) -> float:
    """
    Checks if the provided solution adheres to the deepseek style formatting.
    Accepts solutions that contain <think> and </think> tags followed by a formatted answer.
    The answer does not need to be in <answer> tags.
    """
    pattern = r"^\s*(?:<think>.*?</think>\s*)+.*$"
    formatted = float(bool(re.fullmatch(pattern, solution['solution'], flags=re.DOTALL)))
    if not formatted:
        pass
    return formatted

async def reward_validity( task: Dict, solution: Dict ) -> float:
    """
    Checks to see if the solution matches the ground truth.
    """
    prompt = (
        "Check if the SOLUTION is correct given the GROUND_TRUTH and the TASK"
        f"TASK: {task['row']['problem']}"
        f"SOLUTION: {solution['solution']}"
        f"ANSWER: {task['row']['problem']}"
        "Answer only True or False nothing else"
    )
    answer = await llm.prompt( prompt, model = "unsloth/gemma-3-4b-it" )
    return bool( answer )

async def reward_reasonsing( solution: Dict ) -> bool:
    """
    Checks to see if the reasoning done by the model to produce the outcome is logically consistent.
    """
    prompt_text = (
        "You are an expert logical reasoning auditor. Your task is to examine a chain-of-thought "
        "and determine whether the reasoning presented is genuinely derived from first principles "
        "and builds step by step to the final answer. In particular, evaluate the following criteria:\n\n"
        "1. **Incremental Derivation:** Does each step follow naturally from the previous one, "
        "leading toward the final answer?\n\n"
        "2. **Absence of Pre-Injection:** Is there any indication that the final answer was provided early "
        "in the reasoning or that the subsequent steps are merely a post-hoc justification?\n\n"
        "3. **Logical Consistency:** Are all steps coherent, and do they employ valid logical or mathematical "
        "principles toward deriving the answer?\n\n"
        "Below is the chain-of-thought reasoning along with the final answer. After your evaluation, "
        "please provide *only one word* as your final response: 'true' if the reasoning is logically consistent "
        "and genuinely derived from first principles, or 'false' if it is not.\n\n"
        "Chain-of-Thought Reasoning:\n"
        f"{solution['solution']}"
        "Please analyze the entire reasoning process carefully. "
        "Do not include any additional commentary—only output one word: 'true' or 'false'."
    )
    completion = await llm.prompt(prompt=prompt_text, model = "unsloth/gemma-3-4b-it")
    return float( bool( completion.strip().lower() ) )

def pprompt(text:str, n:int=80) -> str:
    # Replace newlines and ensure total length doesn't exceed n
    text = text.replace('\n', ' ')
    if len(text) <= n:
        return f"[blue]{text}[/blue]"
    else:
        half = (n-3)//2
        return f"[blue]{text[:half]} ... {text[-half:]}[/blue]"   

class Acumen:

    @staticmethod
    def get_config():
        """
        Configures command line arguments for the Acumen script.
        """
        parser = argparse.ArgumentParser(description='Acumen')
        parser.add_argument('--netuid', type=int, default=10)
        parser.add_argument('--dataset', type=str, default="AI-MO/NuminaMath-TIR")
        parser.add_argument('--bucket', type=str, default=comms.bucket())
        parser.add_argument('--window_len', type=int, default=3)
        parser.add_argument('--sample_size', type=int, default=1)
        parser.add_argument('--no_miner', action='store_true', help='Turn off mining')
        parser.add_argument('--no_validator', action='store_true', help='Turn off validating')
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        return bt.config(parser)

    def __init__(self, config):
        """
        Creates runner objects.
        """
        self.console = Console()
        self.config = Acumen.get_config() if config is None else config
        self.console.print(f"\n[bold white]Acumen[/bold white]:\n\n{self.config}")
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.async_subtensor(config=self.config)

    async def initialize(self):
        """
        Initialize objects.
        """
        try:
            self.console.print(f"\n[bold white]Initialization[/bold white]")
            await self.subtensor.initialize()
            self.console.print(f"[light green]\t>> Subtensor:[/light green] {self.subtensor}", style='green')
            self.metagraph = await self.subtensor.metagraph(self.config.netuid)
            self.console.print(f"[light green]\t>> Metagraph:[/light green] {self.metagraph}", style='green')
            if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
                self.console.print(f"[bold red]\tERROR: Hotkey {self.wallet.hotkey.ss58_address} is not registered on subnet {self.config.netuid}, use: `btcli subnet register` ![/bold red]")
                sys.exit()
            self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            self.console.print(f"[light green]\t>> UID:[/light green]{self.uid}", style='green')
            self.bucket = await self.subtensor.get_commitment(netuid=self.config.netuid, uid=self.uid)
            if self.bucket != self.config.bucket:
                self.console.print(f"\tBucket mismatch detected. Updating commitment to: {self.config.bucket}", style='red')
                await self.subtensor.set_commitment(wallet=self.wallet, netuid=self.config.netuid, data=self.config.bucket)
            self.console.print(f"[light green]\t>> Bucket:[/light green] [green]{self.bucket}[/green]", style='green')
            self.weights = None
        except Exception as e:
            self.console.print(f"\tInitialization failed: {str(e)}\n{traceback.format_exc()}", style='bold red')
            sys.exit()
            
    async def window(self) -> int:
        """
        Get the current block window.
        Returns: Current window index
        """
        current_block = await self.subtensor.get_current_block()
        window_value = int(current_block / self.config.window_len)
        return window_value
    
    async def rng_seed(self, window: int) -> str:
        """
        Gets the randomness seed for this window.
        Returns - seed (str): rng seed
        """
        return await self.subtensor.get_block_hash(window * self.config.window_len)
            
    @alru_cache(maxsize=10000)
    async def timestamp( self, window:int ) -> datetime.datetime:
        """
        Get the UTX time of the first block of this window.
        Args:
            window: Block window index
        Returns:
            Timestamp for the block
        """
        try:
            time =  await self.subtensor.get_timestamp( block = window * self.config.window_len )
            return time
        except:
            return datetime.datetime.now()

    @alru_cache(maxsize=128)
    async def get_buckets(self, window: int) -> Dict[ str, str ]:
        """
        Retrieves bucket information for all peers at a specific window.
        Args: window: Block window index
        Returns: Dict mapping hotkey to R2 bucket
        """
        commitments = await self.subtensor.get_all_commitments(netuid=self.config.netuid, block=window * self.config.window_len)
        buckets = {k: v for k, v in commitments.items() if isinstance(v, str) and len(v) == 32 and v.isalnum()}
        return buckets
     
    
    async def mine(self, window:int):
        """
        Solves tasks for window and uploads them to R2.
        Args - window (int): The window to attain tasks for.
        """
        self.console.print(f"\n[bold white]Mining[/bold white] (window={window})", style='yellow')             
        try:
            # Get the rng seed and task list for this window.
            seed = await self.rng_seed(window)
            tasks = await data.get_samples(self.config.dataset, seed=seed, sample_size=self.config.sample_size)
            
            # Function for solving the task using deepseek.
            # NOTE: miners should implement as you see fit.
            async def solve_task_i(i, task):
                prob = task['row']['problem']
                self.console.print(f"\t>> [light green]Solving[/light green] task=({i+1}|{self.config.sample_size}): {pprompt(prob)}", end="\n", style='green')                
                sol_i = await llm.prompt(prompt=prob)#, model="deepseek-ai/DeepSeek-R1")
                signature = self.wallet.hotkey.sign(data=f"{self.wallet.hotkey.ss58_address}{prob}").hex()
                self.console.print(f"\t\t>> [light green]Done[/light green] task=({i+1}|{self.config.sample_size}): {pprompt(sol_i)}", end="\n", style='green')
                return {
                    'idx': i,
                    'row_idx': task['row_idx'],
                    'window': window,
                    'dataset': self.config.dataset,
                    'solution': sol_i,
                    'hotkey': self.wallet.hotkey.ss58_address,
                    'signature': signature
                }
            # Gather solutions.
            solutions = await asyncio.gather(*[solve_task_i(i, task) for i, task in enumerate(tasks)])
            
            # Upload solutions to my bucket for this window.
            filename = f"{self.wallet.hotkey.ss58_address}-solutions-{window}.json"
            await comms.upload(bucket=self.bucket, filename=filename, data=solutions)
            self.console.print(f"\t>> [light green]Uploaded[/light green] {len(tasks)} solutions for window={window}", style='tan')

        except Exception as e:
            self.console.print(f"\t>> ERROR during solve: {str(e)}\n{traceback.format_exc()}", style='red')
            
            
    async def validate(self, window: int):
        """
        Validate solutions from all miners at window.
        Args - window (int): The window to reward solutions.
        """
        self.console.print(f"\n[bold white]Validating[/bold white] (window={window})", style='yellow')             
        try:
            # Get the rng seed, buckets and tasks for window.
            seed = await self.rng_seed(window)
            buckets = await self.get_buckets(window)
            tasks = await data.get_samples(self.config.dataset, seed=seed, sample_size=self.config.sample_size)
            
            # Iterate over each hotkey and get their solutions for this window.
            n_rewards = 0
            reward_info = [{} for _ in tasks]
            for hotkey_i, bucket_i in buckets.items():
                uid = self.metagraph.hotkeys.index(hotkey_i)
                # Download the solutions for this hotkey at this window.                
                fname:str = f"{hotkey_i}-solutions-{window}.json"
                if not await comms.exists(bucket=bucket_i, filename=fname): 
                    self.console.print(f"\t>> [light red]Validating[/light red] uid={uid} has no solutions for window={window}", end="\n", style='tan')
                    continue
                solutions_i = await comms.download(bucket=bucket_i, filename=fname)
                n_sols = len(solutions_i)
                self.console.print(f"\t>> [light green]Validating[/light green] uid={uid} with {n_sols} solutions", end="\n", style='tan')
                # Iterate over solutions, check their signature and reward them.
                for sol_j in solutions_i:
                    idx = int(sol_j['idx'])
                    task_j = tasks[idx]
                    self.console.print(f"\t\t>> [light green]Validating[/light green] uid={uid} solution=({idx}|{n_sols}) : {pprompt(sol_j['solution'])}", end="\n", style='tan')
                    if not bt.Keypair(sol_j['hotkey']).verify(f"{hotkey_i}{task_j['row']['problem']}", bytes.fromhex(sol_j['signature'])):
                        self.console.print(f"Signature mismatch.", style='bold red')
                        continue
                    format: float = await check_format(sol_j)
                    validity: float = await reward_validity(task_j, sol_j)
                    reasoning: float = await reward_reasonsing(sol_j)
                    final: float = 1 + format * validity * reasoning
                    rewards = {'format': format, 'validity': validity, 'reasoning': reasoning, 'final': final }                    
                    self.console.print(f"\t\t\t>> [light green]Validated[/light green] uid={uid} solution=({idx}|{n_sols}) rewards={str(rewards)}", end="\n", style='tan')
                    reward_info[idx][hotkey_i] = rewards
                    n_rewards += 1
            # Optionally upload the rewards to my R2 bucket.                
            if n_rewards > 0:
                await comms.upload(
                    bucket=self.bucket,
                    filename=f"{self.wallet.hotkey.ss58_address}-rewards-{window}.json",
                    data=reward_info,
                )
                self.console.print(f"\t>> [light green]Uploaded[/light green] {len(reward_info)} rewards for window={window}", style='tan')
            else:
                self.console.print(f"\t>> [light red]No[/light red] rewards for window: {window}", style='tan')

        except Exception as e:
            self.console.print(f"ERROR during reward evaluation: {str(e)}\n{traceback.format_exc()}", style='bold red')
            

    async def weight(self, window:int ):
        """
        Sets weights for rewards from this window.
        Args - window (int): The window to attain weights for.
        """
        self.console.print(f"\n[bold white]Weighting[/bold white] (window={window})", style='yellow')  
        if self.weights is None:
            self.weights = torch.zeros( len(self.metagraph.hotkeys) )
        self.console.print(f"\t>> [light green]Weights[/light green]={list(sorted(self.weights.tolist(), reverse=True))[:10]}", style='tan')
        self.console.print(f"\t>> [light green]Uids[/light green]={[i for _, i in sorted(zip(self.weights.tolist(), range(len(self.weights))), reverse=True)][:10]}", style='tan')
        try:
            # Returns the weights for window.
            async def get_grpo_scores_for_window( window: int ) -> List[float]:
                # Check if there is reward info for this window.
                scores = torch.zeros( self.metagraph.n )
                fname = f"{self.wallet.hotkey.ss58_address}-rewards-{window}.json"
                if not await comms.exists(bucket=self.bucket, filename=fname):
                    self.console.print(f"\t>> [light red]Empty[/light red] rewards for window={window}", style='tan')
                    return scores
                # Download the reward info.
                reward_info: List[Dict] = await comms.download(bucket=self.bucket,filename=fname)
                if len(reward_info) == 0:
                    self.console.print(f"\t>> [light red]No[/light red] rewards for window={window}", style='tan')
                    return scores
                
                # Iterate across reward info and add GRPO scores.
                for i, results in enumerate(reward_info):
                    rewards = torch.tensor( [ r['final'] for r in list(results.values()) ], dtype = torch.float32)
                    rewards = grpo_scores(rewards)
                    for hotkey, reward in zip(results.keys(), rewards):
                        idx = self.metagraph.hotkeys.index(hotkey)
                        scores[idx] += reward
                self.console.print(f"\t>> [light green]Generated[/light green] weights for window={window}", style='tan')
                return scores

            # GRPO the weights and set them on chain for the window
            scores: torch.FloatTensor = await get_grpo_scores_for_window( window )
            scores: torch.FloatTensor = scores / (torch.sum(scores) + 1e-8)  # Add small epsilon to avoid division by zero
            self.weights: torch.FloatTensor = 0.01 * scores + (1-0.01) * self.weights.clone().detach()
            await self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=self.metagraph.uids,
                weights=self.weights.tolist(),
                wait_for_inclusion=False
            )
        except Exception as e:
            self.console.print(f"\t>> ERROR during weight setting: {str(e)}\n{traceback.format_exc()}", style='bold red')

    async def run(self):
        """
        Main runner loop.
        """
        # Run forever.
        while True:
            # Load current window and get latest metagraph.
            current_window = await self.window()
            self.metagraph = await self.subtensor.metagraph(self.config.netuid)
            await self.mine(current_window)
            await self.validate(current_window - 1)
            await self.weight(current_window - 2)
            # Wait for window to end.
            while await self.window() == current_window:
                time.sleep(2)
                
# Main entry point.
async def main():
    """
    Initialize and run the pipeline.
    """
    acumen = Acumen(Acumen.get_config())
    await acumen.initialize()
    await acumen.run()

# Run with asyncio.
if __name__ == "__main__":
    asyncio.run(main())