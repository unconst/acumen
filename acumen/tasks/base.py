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

import uuid
import bittensor as bt
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional

def pprompt(text:str, n:int=80) -> str:
    # Replace newlines and ensure total length doesn't exceed n
    text = text.replace('\n', ' ')
    if len(text) <= n:
        return f"[blue]{text}[/blue]"
    else:
        half = (n-3)//2
        return f"[blue]{text[:half]} ... {text[-half:]}[/blue]"   

class DotDict(BaseModel):
    class Config:
        extra = "allow" 

class Task(BaseModel):
    task: DotDict 
    ground_truth: DotDict
    hotkey: Optional[str] = None
    signature: Optional[str] = None
    uuid: str = None
        
    class Config:
        arbitrary_types_allowed = True  # Allow nested dict access
        allow_population_by_field_name = True

    def sign(self, wallet) -> object:
        self.hotkey = wallet.hotkey.ss58_address
        self.signature = wallet.hotkey.sign(data=self.uuid).hex()
        return self
    
    def has_valid_signature(self) -> bool:
        if self.hotkey is None:
            return False
        if self.signature is None:
            return False
        return bt.Keypair(self.hotkey).verify(self.uuid, bytes.fromhex(self.signature))
    
class Solution(BaseModel):
    solution: DotDict
    hotkey: Optional[str] = None
    signature: Optional[str] = None
    uuid: str = None

    class Config:
        arbitrary_types_allowed = True  # Allow nested dict access
        allow_population_by_field_name = True
        
    def sign( self, wallet ) -> object:
        self.hotkey = wallet.hotkey.ss58_address
        self.signature = wallet.hotkey.sign(data=self.uuid).hex()
        return self
    
    def has_valid_signature( self ) -> bool:
        if self.hotkey == None: 
            return False 
        if self.signature == None:
            return False
        return bt.Keypair( self.hotkey ).verify( self.uuid, bytes.fromhex(self.signature))
    
    
class Rewards(BaseModel):
    task: Task
    solution: Solution
    rewards: DotDict
    hotkey: Optional[str] = None
    signature: Optional[str] = None
    uuid: str = None

    def sign( self, wallet ) -> object:
        self.hotkey = wallet.hotkey.ss58_address
        self.signature = wallet.hotkey.sign(data=self.uuid).hex()
        return self
    
    def has_valid_signature( self ) -> bool:
        if self.hotkey == None: 
            return False 
        if self.signature == None:
            return False
        return bt.Keypair( self.hotkey ).verify( self.uuid, bytes.fromhex(self.signature))
    
    
class BaseTask:
    
    def __init__(self):
        pass
    
    async def generate( self, wallet: bt.Wallet ) -> Task:
        task = await self.generate_task()
        task.uuid = str(uuid.uuid4())
        task.sign( wallet )
        return task

    async def generate_task( self ) -> Task:
        """
        Abstract method that should be implemented by subclasses to generate a task.
        Returns a Task object.
        """
        raise NotImplementedError("Subclasses must implement generate_task()")
    
    async def solve(self, wallet: bt.Wallet, task: Task) -> Solution:
        solution = await self.solve_task( task )
        solution.uuid = task.uuid
        solution.sign( wallet )
        return solution

    async def solve_task( self, task: Task ) -> Solution:
        """
        Abstract method that should be implemented by subclasses to solve a task.
        Args:
            task: Dictionary containing the task details
        Returns:
            Dictionary containing the solution
        """
        raise NotImplementedError("Subclasses must implement solve_task()")
    
    async def reward(self, wallet: bt.Wallet, task: Task, solution) -> Rewards:
        rewards = await self.reward_task( task, solution )
        rewards.uuid = task.uuid
        rewards.sign( wallet )
        return rewards
    
    async def reward_task( self, task: Task, solution: Solution ) -> Rewards:
        """
        Abstract method that should be implemented by subclasses to reward a task.
        Args:
            task: Task schema containing task dict and optional ground truth data.
            solution: Solution schema containing task solution dict.
        Returns:
            Dictionary mapping reward names to float values
        """
        raise NotImplementedError("Subclasses must implement reward_task()")
    