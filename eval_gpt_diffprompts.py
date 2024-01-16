from tqdm import tqdm
import pickle
import json
import os
import tiktoken

from log import *
from chat_with_LLM import *

logger = log(__name__).get_log_obj()


def preprocess_data(MCMD_path, language, split="random"):
    filtered_folder_path = os.path.join(MCMD_path, f"filtered_data/{language}/sort_{split}_train80_valid10_test10")
    with open(os.path.join(filtered_folder_path, "test.msg.txt")) as f:
        ref_msg_list = f.read().strip().split("\n")
    with open(os.path.join(filtered_folder_path, "test.sha.txt")) as f:
        sha_list = f.read().strip().split("\n")
    with open(os.path.join(filtered_folder_path, "test.repo.txt")) as f:
        repo_list = f.read().strip().split("\n")
    logger.info(f"There is {len(ref_msg_list)} test messages, {len(sha_list)} shas, {len(repo_list)} repos in ({language}).")

    raw_folder_path = os.path.join(MCMD_path, "raw_data", language)
    repo_info_dict = {}
    full_info_list = []
    for repo, sha, msg in tqdm(zip(repo_list, sha_list, ref_msg_list)):
        each_info = {"repo": repo, "sha": sha, "msg": msg}
        if repo not in repo_info_dict.keys():
            repo_info_dict[repo] = {}
            with open(os.path.join(raw_folder_path, f"{repo}.pickle"), "rb") as f:
                repo_data = pickle.load(f)
                for each_commit in repo_data:
                    repo_info_dict[repo][each_commit["sha"]] = each_commit
        each_info["diff"] = repo_info_dict[repo][sha]["diff"]
        each_info["raw_msg"] = repo_info_dict[repo][sha]["msg"]
        full_info_list.append(each_info)
    logger.info(f"There is {len(full_info_list)} full info in ({language}).")

    save_file_path = os.path.join(filtered_folder_path, f"full_info_list.json")
    with open(save_file_path, "w") as f:
        json.dump(full_info_list, f)
    logger.info(f"Saved full info in ({save_file_path}).")

def gen_msg(full_info_list, pl, model_name, sample_num=50, max_token_code_change=4000, keep_exist_part = False):
    # To get the tokeniser corresponding to a specific model in the OpenAI API:
    os.makedirs(os.path.join("experimental_results", model_name, pl), exist_ok=True)
    
    generated_result = dict()
    system_prompt = f"You are a commit message generator for the {pl} repository."
    # if model_name == "starchat-beta":
        # inputs = tokenizer(system_prompt, return_tensors="pt").to('cuda') # default: GPU
        # system_prompt_token_num = len(inputs.input_ids[0])

    user_prompt_template_list = list()
    user_prompt_template_list.append("I will give you a code change from the {pl} repository and you tell me its commit message.\nThe code change is\n```diff\n{code_change}\n```")
    user_prompt_template_list.append("I will provide you with a code change from the {pl} repository, and you're to generate its commit message. The code change will be presented below:\n```diff\n{code_change}\n```Please respond with the commit message in a single sentence.")
        

    temp_file_path = os.path.join("experimental_results", model_name, pl, f"gen_msg_in_{len(user_prompt_template_list)}_prompts.json")
    exist_idx_list = list()
    if keep_exist_part and os.path.exists(temp_file_path):
        with open(temp_file_path, "r") as f:
            generated_result = json.load(f)
        exist_idx_list = list(generated_result.keys())
        for idx in tqdm(generated_result.keys()):
            if not isinstance(generated_result[idx], str):
                exist_idx_list.remove(idx)
            elif len(generated_result[idx].strip()) == 0:
                exist_idx_list.remove(idx)
        logger.info(f"Loaded {len(exist_idx_list)} generated results from {temp_file_path}.")
        logger.info(f"Examples: {exist_idx_list[:5]}")
    to_do_list = list(set([str(i) for i in range(sample_num)]).difference(set(exist_idx_list)))
    logger.info(f"TODO:{to_do_list[:10]}")
    for idx, each_info in enumerate(tqdm(full_info_list[:sample_num])):
        if keep_exist_part and str(idx) in exist_idx_list:
            continue
        code_change = each_info["diff"]
        # user_prompt = f"I will give you a code change from the {pl} repository and you tell me its commit message. The output format is one sentence.\nThe code change is\n```diff\n{code_change}\n```"
        generated_result[idx] = dict()
        for jdx, user_prompt_template in enumerate(user_prompt_template_list):
            user_prompt = user_prompt_template.format(pl=pl, code_change=code_change)

            # truncate: gpt-3.5-turbo-0613 is limited to 4096 tokens, so we truncate the code change to 4000 tokens
            disallowed_special=None
            if '<|endoftext|>' in code_change:
                disallowed_special=(enc.special_tokens_set - {'<|endoftext|>'})
            length_code_change = len(enc.encode(code_change, disallowed_special=disallowed_special))
            if length_code_change > max_token_code_change:
                code_change = enc.decode(enc.encode(code_change, disallowed_special=disallowed_special)[:max_token_code_change])
                # user_prompt = f"I will give you a code change from the {pl} repository and you tell me its commit message. The output format is one sentence.\nThe code change is\n```diff\n{code_change}\n```"
                user_prompt = user_prompt_template.format(pl=pl, code_change=code_change)
            
            if model_name == "gpt-3.5-turbo-0613":
                try:
                    if str(idx) == to_do_list[0]:
                        logger.info(f"USER PROMPT\n{user_prompt}")
                    gen_msg = chat(system_prompt, user_prompt, model_name)
                except:
                    logger.error(str(idx))
            elif "starchat-beta" in model_name:
                # truncate: starchat is limited to 8192 tokens, so we truncate the code change to 8000 tokens
                # truncate: starchat is limited to 4096 tokens, so we truncate the code change to 4000 tokens
                # inputs = tokenizer(user_prompt, return_tensors="pt").to('cuda') # default: GPU
                # token_num = len(inputs.input_ids[0])
                # if token_num > 4000 - system_prompt_token_num:
                #     user_prompt = tokenizer.decode(inputs.input_ids[0][:4000 - system_prompt_token_num])
            
                prompt_template = "<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>"
                prompt = prompt_template.format(system_prompt=system_prompt, user_prompt=user_prompt)
                
                # We use a special <|end|> token with ID 49155 to denote ends of a turn
                try:
                    outputs = pipe(prompt, max_new_tokens=128, eos_token_id=49155) # do_sample=True, temperature=0.2, top_k=50, top_p=0.95, 
                    # You can sort a list in Python by using the sort() method. Here's an example:\n\n```\nnumbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]\nnumbers.sort()\nprint(numbers)\n```\n\nThis will sort the list in place and print the sorted list.
                    gen_msg = outputs[0]['generated_text'].split("<|assistant|>")[1].strip()
                except:
                    tokenizer = AutoTokenizer.from_pretrained(f"../{model_name}")
                    inputs = tokenizer(user_prompt, return_tensors="pt").to('cuda') # default: GPU
                    token_num = len(inputs.input_ids[0])
                    logger.error(f"idx: {idx}; token_num: {token_num}")
                if idx < 5:
                    logger.info(f"gen_msg:\n{gen_msg}")
            generated_result[idx][jdx] = gen_msg
        if idx % 100 == 0:
            with open(temp_file_path, "w") as f:
                json.dump(generated_result, f)
    return generated_result


if __name__ == '__main__':
    # Step 1: preprocess data
    MCMD_path = "../MCMD"
    # for lan in [ "csharp", "cpp"]:
        # preprocess_data(MCMD_path, lan, split="random")

    # Step 2: generate commit message
    # pl = "java" #" #"
    generated_result_dict = {}
    for pl in ["javascript", "csharp", "cpp", "python", "java"]:
        with open(os.path.join(MCMD_path, f"filtered_data/{pl}/sort_random_train80_valid10_test10/full_info_list.json"), "r") as f:
            full_info_list = json.load(f)
            
        model_name="gpt-3.5-turbo-0613" #"starchat-beta"#
        # if model_name == "gpt-3.5-turbo-0613":
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        if model_name == "starchat-beta":
            import torch
            from transformers import pipeline, AutoTokenizer
            # tokenizer = AutoTokenizer.from_pretrained(f"../{model_name}")
            pipe = pipeline("text-generation", model=f"../{model_name}", torch_dtype=torch.bfloat16, device_map="auto")
        generated_result_dict[pl] = gen_msg(full_info_list, pl, model_name=model_name, keep_exist_part=True)

    gen_file_path = os.path.join("experimental_results", model_name, "all_gen_msg_in_diff_prompts.json")
    with open(gen_file_path, "w") as f:
        json.dump(generated_result_dict, f)