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
import acumen as ac
import bittensor as bt
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional

SCHEMA_VERSION: int = 0 

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
        
    def __str__(self) -> str:
        # Build string of key=value pairs
        result = []
        # Get dict representation of the model
        d = self.dict()
        for k,v in d.items():
            pair = f"{k}={v}"
            # Format this pair with pprompt, ensuring we leave room for ", " between items
            formatted = pprompt(pair, 40) 
            result.append(formatted)
        return ", ".join(result)

class Task(BaseModel):
    type: str = 'NotSet'
    uuid: str = None
    tuid: str = None
    task: DotDict 
    ground_truth: DotDict
    hotkey: Optional[str] = None
    signature: Optional[str] = None
    version: int = SCHEMA_VERSION
    
    def __str__(self):
        return f"Task( type={self.type}, uuid={self.uuid[:5]}, tuid={self.tuid[:5]}, task={self.task})"
    
    def __repr__(self):
        return self.__str__()

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
    type: str = 'NotSet'
    uuid: str = None
    tuid: str = None
    solution: DotDict
    hotkey: Optional[str] = None
    signature: Optional[str] = None
    version: int = SCHEMA_VERSION

    class Config:
        arbitrary_types_allowed = True  # Allow nested dict access
        allow_population_by_field_name = True
        
    def __str__(self):
        return f"Solution( type={self.type}, uuid={self.uuid[:5]}, tuid={self.tuid[:5]}, solution={self.solution})"
        
    def __repr__(self):
        return self.__str__()
    
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
    type: str = 'NotSet'
    uuid: str = None
    tuid: str = None
    task: Task
    solution: Solution
    rewards: DotDict
    hotkey: Optional[str] = None
    signature: Optional[str] = None
    version: int = SCHEMA_VERSION
    
    def __str__(self):
        return f"Rewards( type={self.type}, uuid={self.uuid[:5]}, tuid={self.tuid[:5]}, rewards={self.rewards}, task={self.task.task}), solution={self.solution.solution}))"
    
    def __repr__(self):
        return self.__str__()

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
    
    
class Weights(BaseModel):
    type: str = 'NotSet'
    uuid: str = None
    tuid: str = None
    rewards: List[Rewards] = None
    weights: List[float] = None
    hotkey: Optional[str] = None
    signature: Optional[str] = None
    version: int = SCHEMA_VERSION
    
    def __str__(self):
        return f"Rewards( type={self.type}, uuid={self.uuid[:5]}, tuid={self.tuid[:5]}, weights={self.weights}, rewards={self.rewards}"
    
    def __repr__(self):
        return self.__str__()

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
        try:
            task = await self.generate_task()
        except Exception as e:
            ac.console.print(f"\t>> ERROR during generate_task: {str(e)}", style='red')
            raise e
        task.type = self.__class__.__name__
        task.uuid = str(uuid.uuid4())
        task.tuid = str(uuid.uuid4())
        task.sign( wallet )
        ac.console.print(f"\t>> +{task}", style='green') 
        return task

    async def generate_task( self ) -> Task:
        """
        Abstract method that should be implemented by subclasses to generate a task.
        Returns a Task object.
        """
        raise NotImplementedError("Subclasses must implement generate_task()")
    
    async def solve(self, wallet: bt.Wallet, task: Task) -> Solution:
        try:
            solution = await self.solve_task( task )
        except Exception as e:
            ac.console.print(f"\t>> ERROR during solve_task: {str(e)}", style='red')
            raise e
        solution.type = self.__class__.__name__
        solution.uuid = str(uuid.uuid4())
        solution.tuid = task.tuid
        solution.sign( wallet )
        ac.console.print(f"\t>> +{solution}", style='green') 
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
        try:
            rewards = await self.reward_task( task, solution )
        except Exception as e:
            ac.console.print(f"\t>> ERROR during reward_task: {str(e)}", style='red')
            raise e
        rewards.type = self.__class__.__name__
        rewards.uuid = str(uuid.uuid4())
        rewards.tuid = task.tuid
        rewards.sign( wallet )
        ac.console.print(f"\t>> +{rewards}", style='green') 
        return rewards
    
    async def reward_task( self, task: Task, solution: Solution ) -> Rewards:
        """
        Abstract method that should be implemented by subclasses to reward a task.
        Args:
            task: Task schema containing task dict and optional ground truth data.
            solution: Solution schema containing task solution dict.
        Returns:
            rewards: Reward schema
        """
        raise NotImplementedError("Subclasses must implement reward_task()")
    
    
    async def weight( self, wallet: bt.Wallet, rewards: List[Rewards] ) -> Weights:
        try:
            float_weights = await self.weight_rewards( rewards )
        except Exception as e:
            ac.console.print(f"\t>> ERROR during weight_rewards: {str(e)}", style='red')
            raise e
        weights = Weights(
            type = self.__class__.__name__,
            uuid = str(uuid.uuid4()),
            tuid = rewards[0].tuid,
            rewards = rewards,
            weights = float_weights,
        )
        weights.sign( wallet )
        ac.console.print(f"\t>> +{weights}", style='green') 
        return weights
    
    async def weight_rewards( self, rewards: List[Rewards] ) -> List[float]:
        """
        Abstract method that should be implemented by subclasses to weight rewards.
        Args:
            rewards: List of rewards to weight.
        Returns:
            weights: Weight schema.
        """
        raise NotImplementedError("Subclasses must implement weight_rewards()")

        
    