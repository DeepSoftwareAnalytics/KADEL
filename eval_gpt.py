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

def gen_msg(full_info_list, pl, model_name, sample_num=45000, max_token_code_change=4000, keep_exist_part = False):
    # To get the tokeniser corresponding to a specific model in the OpenAI API:
    os.makedirs(os.path.join("experimental_results", "gpt-3.5", pl), exist_ok=True)
    enc = tiktoken.encoding_for_model(model_name)
    generated_result = dict()
    system_prompt = f"You are a commit message generator for the {pl} repository."
    temp_file_path = os.path.join("experimental_results", "gpt-3.5", pl, f"gen_msg.json")
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
        disallowed_special=None
        if '<|endoftext|>' in code_change:
            disallowed_special=(enc.special_tokens_set - {'<|endoftext|>'})
        length_code_change = len(enc.encode(code_change, disallowed_special=disallowed_special))
        if length_code_change > max_token_code_change:
            code_change = enc.decode(enc.encode(code_change, disallowed_special=disallowed_special)[:max_token_code_change])
        user_prompt = f"I will give you a code change from the {pl} repository and you tell me its commit message. The output format is one sentence.\nThe code change is\n```diff\n{code_change}\n```"
        try:
            if str(idx) == to_do_list[0]:
                logger.info(f"USER PROMPT\n{user_prompt}")
            gen_msg = chat(system_prompt, user_prompt, model_name)
        except:
            logger.error(str(idx))
        generated_result[idx] = gen_msg
        if idx % 500 == 0:
            with open(temp_file_path, "w") as f:
                json.dump(generated_result, f)
    return generated_result


if __name__ == '__main__':
    # Step 1: preprocess data
    MCMD_path = "../MCMD"
    # for lan in [ "csharp", "cpp"]:
        # preprocess_data(MCMD_path, lan, split="random")

    # Step 2: generate commit message
    pl = "cpp" #cpp" #java"#script"
    generated_result_dict = {}
    with open(os.path.join(MCMD_path, f"filtered_data/{pl}/sort_random_train80_valid10_test10/full_info_list.json"), "r") as f:
        full_info_list = json.load(f)
        
    generated_result_dict[pl] = gen_msg(full_info_list, pl, model_name="gpt-3.5-turbo-0613", keep_exist_part=True)

    gen_file_path = os.path.join("experimental_results", "gpt-3.5", pl, "gen_msg.json")
    with open(gen_file_path, "w") as f:
        json.dump(generated_result_dict[pl], f)


# You are a commit message generator for the C++ project. I will give you a code change from the C++ repository and you tell me its commit message. The output format is one sentence.
# The code change is
# ```diff
# ```

# {idx:code chaneg, repo_fullname, sha, commit_message_reference, generation}
