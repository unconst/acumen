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

import random
import acumen as ac
from typing import List, Dict, Tuple, Optional
from .base import BaseTask, Task, Solution, Rewards

splits = [
    'abap', 'actionscript', 'ada', 'agda', 'antlr', 'apacheconf', 'api-blueprint', 'apl', 'applescript', 'arc',
    'arduino', 'asciidoc', 'asp', 'aspectj', 'assembly', 'ats', 'augeas', 'autohotkey', 'awk', 'batchfile',
    'bitbake', 'blitzmax', 'bluespec', 'boo', 'brainfuck', 'bro', 'c', 'c#', 'c++', 'c2hs-haskell',
    'capn-proto', 'cartocss', 'ceylon', 'chapel', 'clean', 'clojure', 'cmake', 'coffeescript', 'coldfusion',
    'coldfusion-cfc', 'common-lisp', 'creole', 'crystal', 'csound', 'css', 'csv', 'cucumber', 'cuda', 'cython',
    'dart', 'desktop', 'diff', 'digital-command-language', 'dm', 'dns-zone', 'dockerfile', 'dylan', 'eagle',
    'ecl', 'edn', 'eiffel', 'elixir', 'elm', 'emacs-lisp', 'emberscript', 'erlang', 'f#', 'factor', 'fancy',
    'fish', 'flux', 'forth', 'fortran', 'freemarker', 'g-code', 'gas', 'gdscript', 'genshi', 'gentoo-ebuild',
    'gettext-catalog', 'glsl', 'gnuplot', 'go', 'graphql', 'graphviz-dot', 'groff', 'groovy',
    'groovy-server-pages', 'haml', 'handlebars', 'harbour', 'haskell', 'haxe', 'hcl', 'hlsl', 'html',
    'html+django', 'html+eex', 'html+erb', 'html+php', 'http', 'hy', 'idris', 'igor-pro', 'inform-7', 'ini',
    'inno-setup', 'io', 'ioke', 'isabelle', 'jade', 'jasmin', 'java', 'java-server-pages', 'javascript',
    'jflex', 'json', 'json5', 'jsoniq', 'jsonld', 'jsx', 'julia', 'jupyter-notebook', 'kotlin', 'krl', 'latte',
    'lean', 'less', 'lfe', 'lilypond', 'linker-script', 'liquid', 'literate-agda', 'literate-coffeescript',
    'literate-haskell', 'livescript', 'llvm', 'logos', 'logtalk', 'lsl', 'lua', 'm4', 'makefile', 'mako',
    'maple', 'markdown', 'mask', 'mathematica', 'mediawiki', 'metal', 'mirah', 'modelica',
    'module-management-system', 'monkey', 'moonscript', 'mtml', 'mupad', 'nesc', 'netlinx', 'nginx', 'nimrod',
    'ninja', 'nit', 'nix', 'nsis', 'nu', 'objective-c++', 'ocaml', 'ooc', 'opencl', 'openscad', 'org', 'oz',
    'pan', 'parrot-assembly', 'parrot-internal-representation', 'pascal', 'pawn', 'perl', 'perl6', 'php',
    'piglatin', 'pike', 'pod', 'pony', 'postscript', 'pov-ray-sdl', 'powershell', 'processing',
    'propeller-spin', 'protocol-buffer', 'pure-data', 'purebasic', 'purescript', 'python', 'qmake', 'qml', 'r',
    'racket', 'ragel-in-ruby-host', 'raml', 'rdoc', 'rebol', 'red', 'renpy', 'restructuredtext', 'rhtml',
    'robotframework', 'rouge', 'ruby', 'rust', 'sage', 'saltstack', 'sas', 'sass', 'scala', 'scaml', 'scheme',
    'scilab', 'scss', 'shell', 'slash', 'slim', 'smalltalk', 'smarty', 'smt', 'solidity', 'sourcepawn',
    'sparql', 'sqf', 'sql', 'squirrel', 'standard-ml', 'stata', 'ston', 'stylus', 'supercollider', 'svg',
    'swift', 'systemverilog', 'tcl', 'tcsh', 'tex', 'text', 'textile', 'thrift', 'toml', 'turtle', 'twig',
    'typescript', 'unity3d-asset', 'unknown', 'uno', 'unrealscript', 'urweb', 'vala', 'vcl', 'vhdl', 'viml',
    'visual-basic', 'volt', 'vue', 'webidl', 'wisp', 'xbase', 'xml', 'xpages', 'xproc', 'xquery', 'xs',
    'xslt', 'xtend', 'yacc', 'yaml', 'yang', 'zephir', 'zig'
]

# Template for constructing the change task using commit details.
COMMIT_TASK = """
You are given a commit that updates a code file.
Old file: {old_file}
New file: {new_file}

Modified Original Code:
{old_contents}

Modified Resulting Code:
{new_contents}

Commit message: {message}

Using the above details, generate a task that instructs a developer to change the code accordingly.
Step 1: The "Modified Original Code" is a version of the original file with minor modifications (without altering its logic).
Step 2: The "Modified Resulting Code" applies the same modifications semantically.
Step 3: Your task should combine the commit message and the two modified code snippets so that the intended changes are clear.

Return only the generated task.
"""

COMMIT_VALIDITY_TEMPLATE = """
You are checking if a code change matches the intended modifications.

Original Code:
{original_code}

Ground Truth (Modified Code):
{ground_truth}

Generated Code:
{solution}

Compare the generated code with the ground truth. Focus on whether the key logical changes from the original code have been properly implemented in the generated code, rather than expecting an exact match.

Consider:
1. Are the core functional changes present?
2. Does it solve the same problem in a similar way?
3. Are the key modifications from the original code reflected?

Answer only True if the generated code captures the essential changes, or False if it misses the key modifications.

NOTE only reply True or False.
"""

class CommitsTask( BaseTask ):
    """A task class for handling code commit-related operations."""
        
    def __init__(self):
        self.loader = ac.data.InfiniteAsyncDataLoader(
            "bigcode/commitpackft",
            sample_batch_size=1,
            config = 'python',
            buffer_size=10
        )
        
    async def __del__(self):
        await self.loader.close()

    async def generate_task( self ) -> Task:
        # Get random example from this dataset.
        example = (await self.loader.next())['row']
        
        # Transform the original file content with minor modifications.
        task_for_old = (
            "Rewrite the following code with minor modifications (for example, replacing general concepts with more specific ones) "
            "Do not add any additional text like 'Here is the rewritten code', instead simply return the new code"
            "ONLY RETURN THE NEW CODE."
            "while preserving its original logic:\n\n" + example["old_contents"]
        )
        print (task_for_old)
        modified_old = await ac.llm.prompt(task_for_old)
        
        # Transform the new file content ensuring the same changes are applied semantically.
        task_for_new = (
            "This is how I changed the previous file:\n\n" + modified_old + "\n\n"
            "Do not add any additional text like 'Here is the rewritten code', instead simply return the new code"
            "ONLY RETURN THE NEW CODE."
            "Now rewrite the following code with equivalent modifications (ensuring the same change is applied semantically) "
            "without changing its overall functionality:\n\n" + example["new_contents"]
        )
        modified_new = await ac.llm.prompt(task_for_new)
        
        # Create task from prompt template.
        prompt = COMMIT_TASK.format(
            old_file=example["old_contents"],
            new_file=example["new_contents"],
            old_contents=modified_old.strip(),
            new_contents=modified_new.strip(),
            message=(example["message"].strip())
        )
        
        # Modify the example.
        return Task(
            task = {
                'prompt': prompt,
                'old_contents': modified_old.strip()
            },
            ground_truth = {
                'modified_new': modified_new.strip(),
            }
        )
            
    async def solve_task( self, task: Task ) -> Solution:
        response = await ac.llm.prompt( prompt = task.prompt, model = "deepseek-ai/DeepSeek-R1")
        start = response.find('<think>') + len('<think>')
        end = response.find('</think>')
        reasoning = response[start:end] 
        answer = response[end:] 
        return Solution(
            solution = {
                'reasoning': reasoning,
                'answer': answer
            }
        )
    
    async def reward_task( self, task: Task, solution: Solution ) -> Rewards:
        check_semantic_implementation = COMMIT_VALIDITY_TEMPLATE.format(
            original_code = task.old_contents,
            ground_truth = task.ground_truth.modified_new,
            solution = solution.solution.answer
        )
        # Return the rewards dict.
        return Rewards(
            task = task,
            solution = solution,
            rewards = {
                'matches': await ac.llm.prompt(check_semantic_implementation),
                'reasoning': await ac.rewards.reason( solution = solution.solution.solution )
            }
        )
    