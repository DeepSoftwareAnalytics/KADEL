import re, os, time, json, datetime, logging, argparse, traceback
from sklearn.metrics import *
from tqdm import tqdm
import subprocess
import logging
import pickle

logger = logging.getLogger(__name__)

# check the msg whether is in Angular convention # https://github.com/angular/angular.js/blob/master/DEVELOPERS.md#-git-commit-guidelines
def check_angular_convention(msg, has_space=True, strict=True):
    if has_space:
        pattern = '((chore)|(docs)|(feat)|(fix)|(perf)|(refactor)|(style)|(test))\s(\((\s|\S)+\)\s)?:(\s(\s|\S)+)?'
    else:
        pattern = '((chore)|(docs)|(feat)|(fix)|(perf)|(refactor)|(style)|(test))(\((\s|\S)+\))?:(\s\S+(\s|\S)+)?'
    if not strict:
        pattern = '^((chore)|(docs)|(feat)|(fix)|(perf)|(refactor)|(style)|(test))'
    return re.match(pattern, msg) != None

'''
situation #1.1: <type>(<scope>): <subject>
situation #1.2: <type> ( <scope> ) : <subject>
situation #2.1: <type>: <subject>
situation #2.2: <type> : <subject>
situation #3.1: <subject>
situation #4.1: <type> ( xxxxx :<subject>
'''
def get_content(txt_line, content_type=["subject"], # TODO content_type -> select_content_type
                currect_content_type = ["type", "scope", "subject"],
                txt_scope_none = None, # TODO check "NNNone"
                add_label = None):  # example: add_label = {"type_scope":<type_scope>, "subject":<subject>}; {"type":<type>, "scope":<scope>, "subject":<subject>}; 
    # Check if the line follows the format: <type>(<scope>): <subject>
    if currect_content_type == ["type", "scope", "subject"] or currect_content_type == ["type", "subject"]:
        if check_angular_convention(txt_line, has_space=False): # situation 1.1 & 2.1
            # check if scope in txt_line
            pattern = re.compile(r'((chore)|(docs)|(feat)|(fix)|(perf)|(refactor)|(style)|(test))(\((\s|\S)+\))?:(\s\S+(\s|\S)+)?')
            match = pattern.match(txt_line)
            txt_type = match.group(1)
            txt_scope = match.group(10)
            if match.group(12):
                subject = match.group(12).strip()
            else:
                subject = ""
        elif check_angular_convention(txt_line, has_space=True): # situation 1.2 & 2.2
            pattern = re.compile(r'((chore)|(docs)|(feat)|(fix)|(perf)|(refactor)|(style)|(test))\s(\((\s|\S)+\)\s)?:(\s(\s|\S)+)?')
            match = pattern.match(txt_line)
            txt_type = match.group(1)
            txt_scope = match.group(10)
            if match.group(12):
                subject = match.group(12).strip()
            else:
                subject = ""
            
        elif ":" in txt_line and check_angular_convention(txt_line, has_space=True, strict=False): # situation 4.1 + (TODO subject starts with type and includes ":")
            txt_type = txt_line.split(" ")[0]
            txt_scope = " ".join(txt_line.split(":")[0].split(" ")[1:])
            subject = ":".join(txt_line.split(":")[1:]).strip()
        else: # situation 3.1
            txt_type, txt_scope = None, None
            subject = txt_line.strip()

        if txt_scope is None:
            txt_scope = txt_scope_none

    elif currect_content_type == ["scope", "subject"]:
        txt_type = None
        txt_scope = txt_line.split(":")[0].strip()
        subject = ":".join(txt_line.split(":")[1:]).strip()
        # else:
        #     txt_scope = txt_scope_none
        #     subject = txt_line.strip()

    if currect_content_type == content_type:
        return txt_line
    
    # Format the content list
    content_list = list()
    for each_type in content_type:
        if each_type == "type" and txt_type is not None:
            if add_label is not None:
                 if "type_scope" in add_label.keys():
                    content_list.append(add_label["type_scope"])
                 if "type" in add_label.keys():
                    content_list.append(add_label["type"])
            content_list.append(txt_type)
        elif each_type == "scope" and txt_scope is not None:
            if add_label is not None and "scope" in add_label.keys():
                content_list.append(add_label["scope"])
            content_list.append(txt_scope)
        elif each_type == "subject":
            if add_label is not None and "subject" in add_label.keys():
                content_list.append(add_label["subject"])
            if content_type != ["subject"] and currect_content_type != ["subject"]:
                content_list.append(":")
            if subject is not None:
                content_list.append(subject)
        elif txt_type is None or txt_scope is None:
            if txt_type is None:
                logger.debug("Note that `txt_type` is None in this sentence {}".format(txt_line))
            if txt_scope is None:
                logger.debug("Note that `txt_scope` is None in this sentence {}".format(txt_line))
            continue
        else:
            logger.info("`content_type` must a list and each item is one of them: \"type\", \"scope\", \"subject\" but the `content_type` is {}".format(content_type))
    return " ".join(content_list)


# !pip install nltk==3.6.2 scipy==1.5.2 pandas==1.1.3 krippendorff==0.4.0 scikit-learn==0.24.1 sumeval==0.2.2 sacrebleu==1.5.1 matplotlib==3.5.1
def get_metric_result(ref_path, gen_path, CommitMsgEmpirical_repo_path="./", only_bleu_norm = False):
    results = dict()
    BN_file = os.path.join(CommitMsgEmpirical_repo_path, "metrics/B-Norm.py")
    BN_evaluate_cmd = "python {} {} < {}".format(BN_file, ref_path, gen_path)
    results['B-Norm'] = float(os.popen(BN_evaluate_cmd).read())
    if only_bleu_norm:
        return results
    Rouge_file = os.path.join(CommitMsgEmpirical_repo_path, "metrics/Rouge.py")
    Rouge_evaluate_cmd = "python {} --ref_path {} --gen_path {}".format(Rouge_file, ref_path, gen_path)
    rouge_str = os.popen(Rouge_evaluate_cmd).read().replace("'", "\"")
    rouge_dict = json.loads(rouge_str)
    results['Rouge-1'], results['Rouge-2'], results['Rouge-L'] = rouge_dict['ROUGE-1'], rouge_dict['ROUGE-2'], rouge_dict['ROUGE-L']
    Meteor_file = os.path.join(CommitMsgEmpirical_repo_path, "metrics/Meteor.py")
    Meteor_evaluate_cmd = "python {} --ref_path {} --gen_path {}".format(Meteor_file, ref_path, gen_path)
    results['Meteor'] = float(os.popen(Meteor_evaluate_cmd).read().strip())
    return results

def evaluate(ref_path, gen_path,
             gen_component_list = ["type", "scope", "subject"], # default: our model always generate all components of them
             ref_component_list = ["type", "scope", "subject"], # default: reference includes all components of them
             txt_scope_none = None, #"NNNone", DIFF#01
             del_parentheses_in_scope = False,
             filter_scope_none = False,
             CommitMsgEmpirical_repo_path="./",
             idx_list = None,
             gen_has_prefix_before_tab = False,
             only_bleu_norm = False, gen_path_prefix_idx=False,
             debug=False):
    ## read gen and ref
    with open(gen_path) as gen_f, open(ref_path) as ref_f:
        gen_msg_list = gen_f.read().strip().split("\n")
        if gen_path_prefix_idx:
            gen_msg_list = [idx_msg.split("\t")[1] for idx_msg in gen_msg_list]
        ref_msg_list = ref_f.read().strip().split("\n")
    
    if idx_list is not None:
        idx_list = list(filter(lambda x:x<min(len(gen_msg_list), len(ref_msg_list)),idx_list))
        gen_msg_list = [gen_msg_list[idx] for idx in idx_list]
        ref_msg_list = [ref_msg_list[idx] for idx in idx_list]
    elif len(gen_msg_list) != len(ref_msg_list):
        idx_list = list(filter(lambda x:x<min(len(gen_msg_list), len(ref_msg_list)),list(range(max(len(gen_msg_list), len(ref_msg_list))))))
    
        
    if gen_has_prefix_before_tab:
        gen_msg_list = [msg.strip().split("\t")[1] for msg in gen_msg_list]
        
    ## split <Type>, <Scope> and <Subject>
    gen_content_dict = dict()
    ref_content_dict = dict()
    evaluate_part_list = list(set(gen_component_list).intersection(set(ref_component_list)))
    for evaluate_part in evaluate_part_list:
        gen_content_dict[evaluate_part] = [get_content(msg, content_type = [evaluate_part], \
                                                   currect_content_type=gen_component_list, \
                                                   txt_scope_none = txt_scope_none) for msg in gen_msg_list]
        ref_content_dict[evaluate_part] = [get_content(msg, content_type = [evaluate_part], \
                                                   currect_content_type=ref_component_list, \
                                                   txt_scope_none = txt_scope_none) for msg in ref_msg_list]
        if evaluate_part == "scope":
            if del_parentheses_in_scope:
                gen_content_dict[evaluate_part] = [" ".join(msg.split(" ")[1:-1]) for msg in gen_content_dict[evaluate_part]]
                ref_content_dict[evaluate_part] = [" ".join(msg.split(" ")[1:-1]) for msg in ref_content_dict[evaluate_part]]
            if filter_scope_none:
                before_filter_num = len(ref_content_dict[evaluate_part])
                if len(ref_content_dict[evaluate_part]) < len(gen_content_dict[evaluate_part]):
                    gen_content_dict[evaluate_part] = gen_content_dict[evaluate_part][:len(ref_content_dict[evaluate_part])]
                for idx, ref_scope in enumerate(reversed(ref_content_dict[evaluate_part])):
                    if ref_scope.strip() == txt_scope_none or ref_scope.strip() == "" or ref_scope is None:
                        del_idx = before_filter_num - 1 - idx
                        del gen_content_dict[evaluate_part][del_idx], ref_content_dict[evaluate_part][del_idx]
        
    metric_result = dict()
    ## evaluate <TYPE>
    if "type" in evaluate_part_list:
        acc = accuracy_score(ref_content_dict["type"], gen_content_dict["type"])*100
        prec = precision_score(ref_content_dict["type"], gen_content_dict["type"], average='weighted')*100
        recall = recall_score(ref_content_dict["type"], gen_content_dict["type"], average='weighted')*100
        f1 = f1_score(ref_content_dict["type"], gen_content_dict["type"], average='weighted')*100
        logger.debug("<TYPE>: accuracy:{:.2f} , precision:{:.2f}, recall:{:.2f}, f1_score:{:.2f}".format(acc, prec, recall, f1))
        metric_result["type"] = {"accuracy":acc, "precision":prec, "recall":recall, "f1_score":f1}
    ## evaluate <SCOPE>
    if "scope" in evaluate_part_list:
        acc = accuracy_score(ref_content_dict["scope"], gen_content_dict["scope"])*100
        prec = precision_score(ref_content_dict["scope"], gen_content_dict["scope"], average='weighted')*100
        recall = recall_score(ref_content_dict["scope"], gen_content_dict["scope"], average='weighted')*100
        f1 = f1_score(ref_content_dict["scope"], gen_content_dict["scope"], average='weighted')*100
        logger.debug("<SCOPE>: accuracy:{:.2f} , precision:{:.2f}, recall:{:.2f}, f1_score:{:.2f}".format(acc, prec, recall, f1))
        metric_result["scope"] = {"accuracy":acc, "precision":prec, "recall":recall, "f1_score":f1}
    ## evaluate <SUBJECT>
    if "subject" in evaluate_part_list:
        subject_ref_path = "{}.subject".format(ref_path)
        subject_gen_path = "{}.subject".format(gen_path)
        # if not os.path.exists(subject_gen_path) or not os.path.exists(subject_ref_path): TODO finally used this condition
        with open(subject_ref_path, "w") as subject_ref, open(subject_gen_path, "w") as subject_gen:
            subject_ref.write("\n".join(ref_content_dict["subject"]))
            subject_gen.write("\n".join(gen_content_dict["subject"]))
        if debug:
            print("Reference File:{}".format(subject_ref_path))
            print("Generation File:{}".format(subject_gen_path))
            for idx in range(0, len(gen_content_dict["subject"]), 5000):
                print("Ref:{}\nGen:{}\n".format(ref_content_dict["subject"][idx], gen_content_dict["subject"][idx]))
        metric_result["subject"] = get_metric_result(subject_ref_path, subject_gen_path, CommitMsgEmpirical_repo_path, only_bleu_norm=only_bleu_norm)
    return metric_result

def cut(input_file, out_file, max_len):
    with open(input_file) as inf:
        inf = inf.read().split("\n")
    with open(out_file,"w") as f:
        for sentence in inf:
            if len(sentence) == 0:
                continue
            sentence = sentence.split(" ")
            f.write(" ".join(sentence[:max_len]))
            f.write("\n")
            
def get_sub_task_list(language_list=["javascript"], \
                  setting_list=["type_scope_diff__subject", "type_diff__subject", \
                            "diff__subject", "scope_diff__subject", \
                            "raw", "diff__type_subject","diff__scope_subject", "diff_type_NNNone_subject", \
                               "type_scope_diff__type_scope_subject", "type_diff__type_subject", "scope_diff__scope_subject"]):
    sub_tasks = list()
    for lan in language_list:
        for setting in setting_list:
            sub_tasks.append("{}/{}".format(lan, setting))
    return sub_tasks

def main():

    split_type_list = ["train", "valid", "test"]

    if not os.path.exists(args.MCMD_data_folder_path):
        logger.error("The MCMD folder path is not exist.")
        exit()

    os.makedirs(args.data_folder_path, exist_ok=True)

    ## Get indexes of commits which match the template(`<type>(<scope>):<subject>`)
    match_template_idx_list_dict = dict()
    for split_type in split_type_list:
        match_template_idx_list_dict[split_type] = list()
        file_path = os.path.join(args.data_folder_path, "{}.msg.txt".format(split_type))
        if not os.path.exists(file_path):
            cut(os.path.join(args.MCMD_data_folder_path, "{}.diff.txt".format(split_type)), os.path.join(args.data_folder_path, "{}.diff.txt".format(split_type)), args.max_diff_len)
            cut(os.path.join(args.MCMD_data_folder_path, "{}.msg.txt".format(split_type)), file_path, args.max_msg_len)
        with open(file_path) as f:
            msg_list = f.read().strip().split("\n")
        logger.info("Reading {} commit messages...".format(len(msg_list)))
        for idx, msg in enumerate(msg_list):
            if check_angular_convention(msg):
                match_template_idx_list_dict[split_type].append(idx)
        logger.info("There are {} commit messages which match the template(`<type>(<scope>):<subject>`).".format(len(match_template_idx_list_dict[split_type])))
        with open(os.path.join(args.data_folder_path, "{}.match_template.txt".format(split_type)), "w") as f:
            f.write("\n".join(list(map(lambda x: str(x), match_template_idx_list_dict[split_type]))))

    
if __name__ == '__main__':
    with open(os.sys.argv[0]) as this_file_content_f:
        this_file_content = this_file_content_f.read()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')

    # Use FileHandler to output results to file
    last_modified_time = time.gmtime(os.path.getmtime(os.sys.argv[0]))
    current_time = time.strftime("%H%M%S", time.localtime())
    formatted_last_modified_time = time.strftime("%Y%m%d_%H%M%S",last_modified_time)
    fh = logging.FileHandler("{}_{}-{}.log".format(os.sys.argv[0][:-3], \
                                                   formatted_last_modified_time, current_time))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Use StreamHandler() to the screen
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    
    logger.info("Start...")
    s_time = time.time()
    
    ## Global variables ##

    ## Parameter ##
    parser = argparse.ArgumentParser(description='WHAT')

    parser.add_argument("--programming_language", 
        default="javascript", required = False)

    parser.add_argument("--MCMD_data_folder_path", 
        default="../MCMD/filtered_data/javascript/sort_random_train80_valid10_test10", required = False)
    
    parser.add_argument("--max_diff_len", 
        default=200, required = False)
    
    parser.add_argument("--max_msg_len", 
        default=50, required = False)

    parser.add_argument("--data_folder_path", 
        default="data/cmt_msg_gen/cut_diff200_msg50/javascript", required = False)

    args = parser.parse_args()

    logger.info(vars(args))
    
    ## Main part ##
    try:
        main()
    except Exception as e:
        logging.error("Main program error:")
        logging.error(e)
        logging.error(traceback.format_exc())

    logger.info("Finished.")
    f_time = time.time()
    duration = f_time - s_time
    logger.info("Duration: {}".format(str(datetime.timedelta(seconds=duration)))) # TODO for more than one days
    logger.debug("="*18)
    logger.debug("Source Code Below: \n{}".format(this_file_content))
