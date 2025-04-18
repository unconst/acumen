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

import acumen as ac
from typing import List, Dict, Tuple, Optional
from .base import BaseTask, Task, Solution, Rewards

class MathTask(BaseTask):
    
    DATASET: str = "AI-MO/NuminaMath-TIR"

    async def generate_task( self ) -> Task:
        # Get random example from this dataset.
        example = await ac.data.get_samples( dataset=MathTask.DATASET )        # Generate task and answer
        return Task(
            task = {
                'question': example[0]['row']['problem']
            },
            ground_truth = {
                'solution': example[0]['row']['solution']
            }
        )
            
    async def solve_task( self, task: Task ) -> Solution:
        return Solution(
            solution = {
                'solution': await ac.llm.prompt( prompt = task.task.question )
            }
        )
    
    async def reward_task( self, task: Task, solution: Solution ) -> Rewards:
        return Rewards(
            task = task,
            solution = solution,
            rewards = {
                'validity': await ac.rewards.validity( 
                    ground_truth = task.ground_truth.solution, 
                    solution = solution.solution.solution  
                ),
                'reasoning': await ac.rewards.reason( 
                    solution = solution.solution.solution  
                )
            }
        )
    