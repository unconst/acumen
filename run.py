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
from typing import List, Dict, Tuple, Optional, Union, Any

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
        ac.console.print(f"\n[bold white]Acumen[/bold white]:\n\n{self.config}")
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.async_subtensor(config=self.config)
        self.task = ac.tasks.MathTask()

    async def initialize(self):
        """
        Initialize objects.
        """
        try:
            ac.console.print(f"\n[bold white]Initialization[/bold white]")
            await self.subtensor.initialize()
            ac.console.print(f"[light green]\t>> Subtensor:[/light green] {self.subtensor}", style='green')
            self.metagraph = await self.subtensor.metagraph(self.config.netuid)
            ac.console.print(f"[light green]\t>> Metagraph:[/light green] {self.metagraph}", style='green')
            if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
                ac.console.print(f"[bold red]\tERROR: Hotkey {self.wallet.hotkey.ss58_address} is not registered on subnet {self.config.netuid}, use: `btcli subnet register` ![/bold red]")
                sys.exit()
            self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            ac.console.print(f"[light green]\t>> UID:[/light green]{self.uid}", style='green')
            self.bucket = await self.subtensor.get_commitment(netuid=self.config.netuid, uid=self.uid)
            if self.bucket != self.config.bucket:
                ac.console.print(f"\tBucket mismatch detected. Updating commitment to: {self.config.bucket}", style='red')
                await self.subtensor.set_commitment(wallet=self.wallet, netuid=self.config.netuid, data=self.config.bucket)
            ac.console.print(f"[light green]\t>> Bucket:[/light green] [green]{self.bucket}[/green]", style='green')
            self.weights = None
        except Exception as e:
            ac.console.print(f"\tInitialization failed: {str(e)}\n{traceback.format_exc()}", style='bold red')
            sys.exit()
            
    async def window(self) -> int:
        """
        Get the current block window.
        Returns: Current window index
        """
        current_block = await self.subtensor.get_current_block()
        window_value = int(current_block / self.config.window_len)
        return window_value
                
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
        try:
            # 1) Gets name for next tasks.
            task_fname = f"{self.wallet.hotkey.ss58_address}-tasks-{window}.json"
            # 2) Generates N new tasks fron generator.
            tasks = await asyncio.gather(*[
                self.task.generate(self.wallet) 
                for _ in range(self.config.sample_size)
            ])
            # 3) Upload tasks.
            await ac.comms.upload(bucket=self.bucket, filename=task_fname, data=tasks)
        except Exception as e:
            ac.console.print(f"\t>> ERROR during generation: {str(e)}\n{traceback.format_exc()}", style='red')
            
    async def solve(self, window: int):
        try:
            # 1) Checks for tasks, solves and uploads solutions.
            async def _process_bucket(hotkey_i, bucket_i, window):
                # 2) Checks and downloads tasks.
                fname = f"{hotkey_i}-tasks-{window}.json"
                if not await ac.comms.exists(bucket=bucket_i, filename=fname): return
                tasks = await ac.comms.download(bucket=bucket_i, filename=fname)
                # 3) Solves and uploads solutions.
                solutions = await asyncio.gather(
                    *(self.task.solve(self.wallet, task) for task in tasks),
                    return_exceptions=False
                )
                out_fname = f"{self.wallet.hotkey.ss58_address}-{hotkey_i}-solutions-{window}.json"
                await ac.comms.upload(bucket=self.bucket, filename=out_fname, data=solutions)                
            # 4) Run all solvers.
            buckets = await self.get_buckets(window)
            await asyncio.gather(
                *(_process_bucket(hk, bk, window) for hk, bk in buckets.items()),
                return_exceptions=False
            )
        except Exception as e:
            ac.console.print(f"\t>> ERROR during generation: {str(e)}\n{traceback.format_exc()}", style='red')
            
    async def reward(self, window: int):
        try:
            # 1) Check and download tasks for this window.
            tasks_fname = f"{self.wallet.hotkey.ss58_address}-tasks-{window}.json"
            if not await ac.comms.exists(bucket=self.bucket, filename=tasks_fname): return
            tasks = await ac.comms.download(bucket=self.bucket, filename=tasks_fname)

            # 2) Downloads solutions and rewards them in parallel.
            async def _process_rewards(hotkey_i, bucket_i, window):
                # Check and download solutions.
                solution_fname = f"{hotkey_i}-{self.wallet.hotkey.ss58_address}-solutions-{window}.json"
                if not await ac.comms.exists(bucket=bucket_i, filename=solution_fname): return
                solutions = await ac.comms.download(bucket=bucket_i, filename=solution_fname)

                # Generate rewards and upload them for each.
                rewards = await asyncio.gather(
                    *(self.task.reward(self.wallet, task, sol)
                    for task, sol in zip(tasks, solutions)),
                    return_exceptions=False
                )
                rewards_fname = f"{self.wallet.hotkey.ss58_address}-{hotkey_i}-rewards-{window}.json"
                await ac.comms.upload(bucket=self.bucket, filename=rewards_fname, data=rewards)
                
            # 3) Run all rewarders.             
            buckets = await self.get_buckets(window)
            await asyncio.gather(
                *(_process_rewards(hk, bk, window) for hk, bk in buckets.items()),
                return_exceptions=False
            )            
        except Exception as e:
            ac.console.print(f"\t>> ERROR during generation: {str(e)}\n{traceback.format_exc()}", style='red')
            
    async def set_weights(self, window: int):
        try:
            # 1) Download my tasks for this window (or exit if none)
            tasks_fname = f"{self.wallet.hotkey.ss58_address}-tasks-{window}.json"
            if not await ac.comms.exists(bucket=self.bucket, filename=tasks_fname):return
            tasks = await ac.comms.download(bucket=self.bucket, filename=tasks_fname)
            
            # 2) Gather all rewards by task UUID
            buckets = await self.get_buckets(window)
            all_rewards: dict[str, dict[str, Any]] = {t.tuid: {} for t in tasks}
            for hotkey_i, _ in buckets.items():
                # 3) Check for rewards for this hotkey download them.
                rewards_fname = f"{self.wallet.hotkey.ss58_address}-{hotkey_i}-rewards-{window}.json"
                if not await ac.comms.exists(bucket=self.bucket, filename=rewards_fname): continue
                rewards = await ac.comms.download(bucket=self.bucket, filename=rewards_fname)
                for r in rewards:
                    all_rewards[r.tuid][hotkey_i] = r
                    
            # 3) Weight reward sets.
            all_weights = []
            chain_weights = [ 0 for _ in self.metagraph.hotkeys]
            for _, hotkey_rewards in all_rewards.items():
                if len(hotkey_rewards) == 0: continue 
                weights = await self.task.weight( self.wallet, [ reward for _, reward in hotkey_rewards.items() ] )
                for reward_i, weight_val in zip(weights.rewards, weights.weights):
                    idx = self.metagraph.hotkeys.index(reward_i.solution.hotkey)
                    chain_weights[idx] += weight_val
                all_weights.append( weights )
                    
            # Set weights on chain.
            await self.subtensor.set_weights(
                wallet = self.wallet,
                netuid = self.config.netuid,
                weights=chain_weights,
                uids = self.metagraph.uids
            )
            # Upload weights.
            weights_fname = f"{self.wallet.hotkey.ss58_address}-{hotkey_i}-weights-{window}.json"
            await ac.comms.upload(bucket=self.bucket, filename=weights_fname, data=all_weights)
        except Exception as e:
            ac.console.print(f"\t>> ERROR during generation: {str(e)}\n{traceback.format_exc()}", style='red')
    
    async def run(self):
        """
        Main runner loop.
        """
        # Run forever.
        while True:
            # Load current window and get latest metagraph.
            current_window = await self.window()
            self.metagraph = await self.subtensor.metagraph(self.config.netuid)
            ac.console.print(f"\n[bold white]Window[/bold white] (window={current_window})", style='yellow')             
            await asyncio.gather(
                self.generate(current_window),
                self.solve(current_window - 1),
                self.reward(current_window - 2),
                self.set_weights(current_window - 3)
            )
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