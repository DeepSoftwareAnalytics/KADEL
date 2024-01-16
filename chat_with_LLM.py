from log import *
import hashlib

import redis
import json
import openai


logger = log(__name__).get_log_obj()

with open("secret_api.txt", "r") as f:
    API_TOKEN = f.read().strip()

    
def request_api(model_name, user_prompt, system_prompt, channel = "openai", 
                account_id = None, gateway=None):
    if channel == "openai":
        openai.api_key = API_TOKEN
        return openai.ChatCompletion.create(model=model_name,
                                            messages=[{"role": "system", "content": system_prompt},
                                                      {"role": "user", "content": user_prompt},])
    elif channel  == "cloudflare":
        import requests
        API_BASE_URL = f"https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway}/openai/chat/completions"
        headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
        data = {"model": model_name,
                "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": user_prompt}]}
        response = requests.post(API_BASE_URL, headers=headers, json=data)
        return json.loads(response.text)
    
def atomic_chat(system_prompt, user_prompt, max_times=5, model_name="gpt-3.5-turbo-0613", use_cache=False): # TODO random_seed -> cache_key
    if use_cache:
        r = redis.StrictRedis(host='localhost', port=6379, db=0)
        all_prompt = model_name+system_prompt+user_prompt
        cache_key = hashlib.sha256(all_prompt.encode()).hexdigest()
        cached_response = r.get(cache_key)#.decode("utf-8")
        if cached_response:
            logger.debug(json.loads(cached_response))
            return json.loads(cached_response)
    try_times = 0
    tmp_model_name = model_name
    while try_times < max_times:
        if try_times > 0:
            logger.debug(f"Try again... ({try_times+1}-th time)")
        try:
            response = request_api(tmp_model_name, user_prompt, system_prompt)
            if use_cache:
                r.setex(cache_key, 60*60*24*365, json.dumps(response))
            break
        except Exception as e:
            logger.error(f"{try_times+1}-th time Error:{e}")
            if isinstance(e, openai.error.InvalidRequestError) and e.code == "context_length_exceeded":
                token_num = int(e._message.split("However, your messages resulted in ")[1].split(" tokens.")[0])
                logger.error(f"Token number: {token_num}")
                if model_name == "gpt-3.5-turbo-0613":
                    tmp_model_name = "gpt-3.5-turbo-16k-0613"
            else:
                time.sleep(30)
            try_times += 1
    return response

def chat(system_prompt, user_prompt, model_name="gpt-3.5-turbo-0613", output_format = None, use_cache=False, remove_quote=False):
    if "gpt-3.5" in model_name or "gpt-4" in model_name:
        response = atomic_chat(system_prompt, user_prompt, model_name=model_name, use_cache=use_cache)
        logger.debug(response)

        if 'choices' in response.keys():
            response_txt = response['choices'][0]['message']['content']
        else:
            return response
        
        response_txt = response['choices'][0]['message']['content']
        if output_format is not None:
            check_system_prompt = "You are an Information Extraction Expert. You check if the output strictly follows the format."
            check_user_prompt = f"Please determine whether the following Content (between '```' and '```') meets the Format. If so, output just 'True'. Otherwise, extract the content part that strictly follows the Format from the Content as output.\nContent:```{response_txt}```\nFormat:{output_format}"
            check_response = atomic_chat(check_system_prompt, check_user_prompt, model_name=model_name, use_cache=use_cache)
            if not check_response['choices'][0]['message']['content'] == "True":
                response_txt = check_response['choices'][0]['message']['content']

        if remove_quote:
            if (response_txt[0] == "\'" and response_txt[-1] == "\'") or \
                (response_txt[0] == "\"" and response_txt[-1] == "\""):
                response_txt = response_txt[1:-1]
            if response_txt == "":
                response_txt = "<NONE>"
        return response_txt