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
import json
import sys
import time
import torch
import asyncio
import datetime
import argparse
import traceback
import acumen as ac
import bittensor as bt
from async_lru import alru_cache
from rich.console import Console
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

class Acumen:

    @staticmethod
    def get_config():
        """
        Configures command line arguments for the Acumen script.
        """
        parser = argparse.ArgumentParser(description='Acumen')
        parser.add_argument('--netuid', type=int, default=10)
        parser.add_argument('--dataset', type=str, default="AI-MO/NuminaMath-TIR")
        parser.add_argument('--bucket', type=str, default=ac.comms.bucket())
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
        self.task = ac.tasks.MathTask()

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
    
    async def generate(self, window: int):
        self.console.print(f"\n[bold white]Generating Tasks[/bold white] (window={window})", style='tan')             
        try:
            tasks = []
            for _ in range(self.config.sample_size):
                task_i = await self.task.generate( self.wallet )
                tasks.append( task_i)
                self.console.print(f"\n\t >> task:{task_i}", style='tan') 
            filename = f"{self.wallet.hotkey.ss58_address}-tasks-{window}.json"
            await ac.comms.upload(bucket=self.bucket, filename=filename, data=tasks)

        except Exception as e:
            self.console.print(f"\t>> ERROR during generation: {str(e)}\n{traceback.format_exc()}", style='red')
            
    async def solve(self, window: int):
        self.console.print(f"\n[bold white]Solving Tasks[/bold white] (window={window})", style='yellow')             
        try:
            buckets = await self.get_buckets(window)
            for hotkey_i, bucket_i in buckets.items():
                fname:str = f"{hotkey_i}-tasks-{window}.json"
                if not await ac.comms.exists(bucket=bucket_i, filename=fname): 
                    continue
                tasks = await ac.comms.download(bucket=bucket_i, filename=fname) 
                solutions = []
                for _, task in enumerate(tasks):
                    solution_i = await self.task.solve( self.wallet, task ) 
                    solutions.append( solution_i )
                    self.console.print(f"\n\t >> solution={solution_i}", style='tan')             
                # Upload solutions.
                filename = f"{self.wallet.hotkey.ss58_address}-{hotkey_i}-solutions-{window}.json"
                await ac.comms.upload(bucket=self.bucket, filename=filename, data=solutions)
        except Exception as e:
            self.console.print(f"\t>> ERROR during generation: {str(e)}\n{traceback.format_exc()}", style='red')
            
    async def reward(self, window: int):
        self.console.print(f"\n[bold white]Rewarding Tasks[/bold white] (window={window})", style='yellow')             
        try:
            buckets = await self.get_buckets(window)
            tasks_fname:str = f"{self.wallet.hotkey.ss58_address}-tasks-{window}.json"
            # Check if tasks exist for this window.
            if not await ac.comms.exists(bucket=self.bucket, filename=tasks_fname): 
                return
            # Get the tasks I uploaded.
            tasks = await ac.comms.download(bucket=self.bucket, filename=tasks_fname) 
            self.console.print(f"\n\t >> loaded: tasks={len(tasks)}, window={window}", style='tan')             
            # Got through all buckets and finc someone who solved my tasks.
            for hotkey_i, bucket_i in buckets.items():
                # Check if this hotkey solved these tasks.
                solution_fname:str = f"{hotkey_i}-{self.wallet.hotkey.ss58_address}-solutions-{window}.json"
                if not await ac.comms.exists(bucket=self.bucket, filename=solution_fname): 
                    continue
                # Get the uploaded solutions.
                solutions = await ac.comms.download(bucket=bucket_i, filename=solution_fname) 
                # Reward solutions given tasks.
                rewards = []
                for task, sol in list(zip(tasks, solutions)):
                    rewards_i = await self.task.reward( self.wallet, task, sol )
                    rewards.append( rewards_i )
                    self.console.print(f"\n\t\t >> reward={rewards_i}", style='tan')             
                # Upload rewards.
                rewards_fname = f"{self.wallet.hotkey.ss58_address}-{hotkey_i}-rewards-{window}.json"
                await ac.comms.upload(bucket=self.bucket, filename=rewards_fname, data=rewards)
                self.console.print(f"\n\t >> uploaded: rewards={len(rewards)}, window={window}", style='tan')             
        except Exception as e:
            self.console.print(f"\t>> ERROR during generation: {str(e)}\n{traceback.format_exc()}", style='red')
    
    async def run(self):
        """
        Main runner loop.
        """
        # Run forever.
        while True:
            # Load current window and get latest metagraph.
            current_window = await self.window()
            self.metagraph = await self.subtensor.metagraph(self.config.netuid)
            await self.generate(current_window)
            await self.solve(current_window - 1)
            await self.reward(current_window - 2)
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