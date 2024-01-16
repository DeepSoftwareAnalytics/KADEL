## Comparison with ChatGPT


### Code

File path: [KADEL/eval_gpt.py](../../eval_gpt.py)

You shold store your OpenAI api token in file named `secret_api.txt` under the root path (such as `$REPO_DIR/KADEL/secret_api.txt`)

### Prompt

`pl` means programming language such as `java`.

`code_change` means the raw code diff.

System Prompt: 

"You are a commit message generator for the {pl} repository."


- Basic Prompt

"""
I will give you a code change from the {pl} repository and you tell me its commit message.
The code change is
```diff
{code_change}
```
"""


- Basic prompt + Output format

User Prompt:

"""
I will give you a code change from the {pl} repository and you tell me its commit message.
The output format is one sentence.
The code change is
```diff
{code_change}
```
"""

- Rephrased prompt by ChatGPT

User Prompt:

"""
I will provide you with a code change from the {pl} repository, and you're to generate its commit message. The code change will be presented below:
```diff
{code_change}
```
Please respond with the commit message in a single sentence.
"""

### Results

Path: [KADEL/experimental_results/gpt-3.5](../../experimental_results/gpt-3.5)
