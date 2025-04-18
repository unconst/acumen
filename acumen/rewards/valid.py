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
from typing import List, Dict, Tuple, Optional, Union

VALIDITY = """
    You must determine if the SOLUTION is semantically THE SAME as the GROUND_TRUTH.
    Do not evaluate correctness - only check if they match.\n
    GROUND_TRUTH: {ground_truth}\n
    SOLUTION: {solution}\n
    Compare the GROUND_TRUTH and SOLUTION strings match semantically.\n
    Answer ONLY 'True' if they match, or 'False' if there is ANY difference.\n
    No other response is allowed.
"""

async def validity( ground_truth: str, solution: str ) -> float:
    prompt = VALIDITY.format( ground_truth = ground_truth, solution = solution )
    validity = await ac.llm.prompt( prompt = prompt )
    score = float( bool( validity ) )
    return score
