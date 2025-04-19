
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

GEN_PROMPT = """
    You are given a reasoning process. Your job is to diligently read the reasoning and break it into a set of dicernable steps.
    Your reasoning should be written as to explain so that a secondary model could read it and it would help them derive the same answer.

    **Output format is NON-NEGOTIABLE:**
    Reply with **one and only one** JSON object whose top-level key is steps.  
    Absolutely no extra keys, comments, Markdown, or prose are allowed.

    Schema (copy exactly):
    {{
    "steps": [
        {{
        "n": <int>,                     // 1, 2, 3 … consecutive
        "type": "<definition|assumption|algebra|inference|substitution|conclusion>",
        "text": "<statement of this step>",
        "derives_from": [<ints>]        // earlier step numbers; [] allowed
        }}
        // … more steps …
    ],
    }}

    **Example**  – delete this block before answering:
    {{
    "steps": [
        {{ "n": 1, "type": "definition",  "text": "Let f(x)=x^2", "derives_from": [] }},
        {{ "n": 2, "type": "algebra",     "text": "f(2) = 2^2 = 4", "derives_from": [1] }},
        {{ "n": 3, "type": "conclusion",  "text": "Therefore f(2)=4", "derives_from": [2] }}
    ],
    }}

    **Generation rules**

    1. Make sure to include all the steps. ALL ARE IMPORTANT.
    1. Use double quotes around every JSON string; no trailing commas.
    2. Step numbers must start at 1 and increase by 1 with no gaps.
    3. Every non-primitive claim must list its dependencies in `derives_from`.
    4. *Do not* add explanatory text outside the JSON object—if you do, the answer will be rejected.

    Now read the reasoning below and respond **only** with the JSON object:

    REASONING:
    {reasoning}

    Parse this reasoning into multiple steps (usually 5 or more) and return them in the format
"""
VAL_PROMPT = """\
    You are **Trace-Judge-v1**, an uncompromising verifier.

    Below is the raw JSON output produced by Solution-Bot:

    ```json
    {solver_output}
    ```

    ─────────────────  YOUR TASK  ─────────────────
    1. Parse the JSON.  It contains:
    • "problem" : str  
    • "steps"   : list[step]  
    • "answer"  : str  

    Each `step` is  
    {{ "n": int, "type": str, "text": str, "derives_from": list[int] }}

    2. For every step *s* in `"steps"`, decide if it is **logically valid**  
    given the PROBLEM statement plus the steps whose numbers appear  
    in `s["derives_from"]`.  
    - If valid → `"valid": true` and leave `"reason": ""`.  
    - If invalid → `"valid": false` and give a SHORT reason.

    3. After checking all steps, set `"verdict"`:  
    • `"ALL_VALID"`  - only if **every** step is valid **and** the ANSWER  
        is explicitly derived by one or more conclusion steps.  
    • `"INVALID"`    - otherwise.

    ──────────────  REQUIRED OUTPUT FORMAT  ──────────────
    Return **one JSON object**, no extra text, with keys:

    {{
    "step_results": [
        {{ "n": 1, "valid": true,  "reason": "" }},
        {{ "n": 2, "valid": false, "reason": "derives_from cites future step" }},
        …
    ],
    "verdict": "ALL_VALID" | "INVALID"
    }}

    Rules:  
    * Preserve step order.  
    * No prose, Markdown, or comments outside the JSON object.  
    * Treat missing or malformed fields as automatic invalidity.
    * dont output anything but the JSON string, i.e. dont provide a justification or reasoning, just the json string which will be parseable.
    * PAY STRICT ATTENTION TO OUTPUTTING IN THE FORMAT REQUIRED!
""" 
# ----------------------------------------------------------
# (re-use the sanitize_json_string() helper from before)
# ----------------------------------------------------------
import json, re
from typing import Optional, Any

# ── 1. remove ``` fences (with or without lang tag) ──────────────────────
_FENCE_RX = re.compile(
    r"^\s*```[\w-]*\s*(.*?)\s*```[\s]*$", re.DOTALL
)

def _strip_fences(txt: str) -> str:
    m = _FENCE_RX.match(txt.strip())
    return m.group(1) if m else txt

# ── 2. remove <think>…</think> blocks (optional) ────────────────────────
_THINK_RX = re.compile(r"<think>.*?</think>", re.DOTALL)

def _strip_think(txt: str) -> str:
    return _THINK_RX.sub("", txt)

# ── 3. double any illegal backslash escape sequences ────────────────────
_BAD_BS_RX = re.compile(
    r"""
    (\\)                 # single backslash
    (?!                  # NOT followed by…
        ["\\/bfnrt]      #   a valid 1-char JSON escape
      | u[0-9a-fA-F]{4}  #   or \uXXXX
    )
    """,
    re.VERBOSE,
)

def _fix_backslashes(txt: str) -> str:
    return _BAD_BS_RX.sub(r"\\\\", txt)

# ── 4. master helper ─────────────────────────────────────────────────────
def safe_json_loads(raw: str) -> Optional[Any]:
    """
    Parse *raw* into JSON, tolerating:
      • ```json fences
      • <think>…</think> hidden thoughts
      • single backslashes in LaTeX like \frac, \pi, \theta …
    Returns None on failure instead of raising.
    """
    text = _strip_think(raw)
    text = _strip_fences(text).strip()
    text.replace('```json', '')

    # First try plain json.loads
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Second pass: repair stray backslashes
    try:
        text_fixed = _fix_backslashes(text)
        return json.loads(text_fixed)
    except json.JSONDecodeError as e:
        print (text_fixed, e)
        return None 
    
async def sound_reasoning( reasoning: str ) -> float:
    parse_prompt = GEN_PROMPT.format(reasoning = reasoning)    
    steps_output = await ac.llm.prompt( parse_prompt, model = "unsloth/Mistral-Nemo-Instruct-2407")
    steps_json = safe_json_loads(steps_output)
    val_prompt = VAL_PROMPT.format(solver_output=json.dumps(steps_json['steps']))
    val_output = await ac.llm.prompt( val_prompt, model = "unsloth/Mistral-Nemo-Instruct-2407")
    val_json = safe_json_loads(val_output)
    return float(val_json.get("verdict") == "ALL_VALID")

