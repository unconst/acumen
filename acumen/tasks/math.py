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

import torch
import random
import acumen as ac
from datasets import load_dataset
from typing import List, Dict, Tuple, Optional
from .base import BaseTask, Task, Solution, Rewards, Weights

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


class MathTask(BaseTask):
    
    def __init__(self):
        self.ds = load_dataset("AI-MO/NuminaMath-TIR", split="train") 

    async def generate_task( self ) -> Task:
        # Get random example from this dataset.
        example = self.ds[random.randint(0, len(self.ds)-1)]
        return Task(
            task = {
                'question': example['problem']
            },
            ground_truth = {
                'solution': example['solution']
            }
        )
            
    async def solve_task( self, task: Task ) -> Solution:
        response = await ac.llm.prompt( prompt = task.task.question, model = "deepseek-ai/DeepSeek-R1" )
        start = response.find('<think>') + len('<think>')
        end = response.find('</think>')
        reasoning = response[start:end] 
        answer = response[end:] 
        return Solution(
            solution = {
                'solution': answer,
                'reasoning': reasoning
            }
        )
    
    async def reward_task( self, task: Task, solution: Solution ) -> Rewards:
        validity: float = await ac.rewards.validity( 
            ground_truth = task.ground_truth.solution, 
            solution = solution.solution.solution  
        )
        reasoning: float = await ac.rewards.sound_reasoning( 
            reasoning = solution.solution.reasoning  
        )
        return Rewards(
            task = task,
            solution = solution,
            rewards = {
                'validity': validity,
                'reasoning': reasoning
            }
        )
        
    async def weight_rewards( self, rewards: List[Rewards] ) -> List[float]:
        scores = torch.zeros( len(rewards ))
        for i, r in enumerate( rewards ):
            scores[i] = r.rewards.validity * r.rewards.reasoning
        weights = grpo_scores( scores )
        return weights.tolist()
        
