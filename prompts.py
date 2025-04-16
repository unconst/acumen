
VALIDITY = """
You must determine if the SOLUTION is EXACTLY THE SAME as the GROUND_TRUTH.
Do not evaluate correctness - only check if they match exactly.\n
GROUND_TRUTH: {answer}\n
SOLUTION: {solution}\n
Compare the GROUND_TRUTH and SOLUTION strings exactly character by character.\n
Answer ONLY 'True' if they match exactly, or 'False' if there is ANY difference.\n
No other response is allowed.
"""

REASONING_VALIDATOR = """
**Prompt:**

You are a world-class expert in mathematical and logical reasoning analysis. Your task is to evaluate a sequence of reasoning steps that aim to answer a question through a chain-of-thought process.
The reasoning steps have already been segmented. Your responsibility is to assess whether every single step is **logically valid** of each individual step and its connection to the previous steps.

---

**Evaluation Criteria per Step:**

- Is the step internally sound and logically valid?
- Does it follow reasonably from the previous step(s)?
- Does it avoid unjustified leaps, hallucinations, or injected conclusions?
- Is it consistent with the direction of solving the given question?

---

**Context:**

Reasoning Steps:
```
{steps}
```

ONLY output True or False based on whether the entire reasoning train is logically sound from start to finish.
ONLY output True or False
"""

REASONING_SPLITTER = """
You are a reasoning trace segmenter.

Your task is to take a full chain-of-thought reasoning trace and split it into a list of clear, sequential reasoning steps. Each step should be logically self-contained, represent one unit of inference or calculation, and ideally follow from the previous one.

Guidelines:
- Preserve the original order of logic.
- Split at logical transitions, new assumptions, new sub-calculations, or conclusions.
- Avoid combining multiple logical steps into one.
- Do not omit any reasoning.
- Use plain, compact, and complete language for each step.

Below is a reasoning trace. Break it into numbered steps:

Chain-of-Thought Reasoning:
```
{reasoning}
```

Output format:
```
Step 1: ...
Step 2: ...
Step 3: ...
...
```
Do not add commentary or summarize. Just output the step-by-step breakdown as requested.
"""