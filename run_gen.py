# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import os
import ast
import json
import math
import time
import scipy
import wandb
import random
import pickle
import logging
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial


import torch
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument("--task", type=str, required=True,
                        choices=['summarize', 'cmt_msg_gen', 'concode', 'translate', 'refine', 'defect', 'clone', 'multi_task'])
    parser.add_argument("--sub_task", type=str, default='')
    parser.add_argument("--lang", type=str, default='')
    parser.add_argument("--eval_task", type=str, default='')
    parser.add_argument("--model_type", default="codet5", type=str, choices=['roberta', 'bart', 'codet5'])
    parser.add_argument("--add_lang_ids", action='store_true')
    parser.add_argument("--data_num", default=-1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_train_epochs", default=100, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--summary_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--res_dir", type=str, required=True)
    parser.add_argument("--res_fn", type=str, default=None)
    parser.add_argument("--add_task_prefix", action='store_true', help="Whether to add task prefix for t5 and codet5")
    parser.add_argument("--save_last_checkpoints", action='store_true')
    parser.add_argument("--always_save_model", action='store_true')
    parser.add_argument("--do_eval_bleu", action='store_true', help="Whether to evaluate bleu on dev set.")

    ## Required parameters
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="roberta-base", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=200, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=50, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--save_steps", default=-1, type=int, )
    parser.add_argument("--log_steps", default=-1, type=int, )
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")
    # new design and its switch
    parser.add_argument("--setting", default={'known_part':['type','scope'], 'unknown_part': ['subject']}, # TODO default value should be None; 
                        help="choose which types of commit message are used in the training", type=ast.literal_eval)
    parser.add_argument("--max_unknown_part_length", default=None, type=int, # TODO default:None
                        help="The maximum length of unknown part(such as <type> and <scope>) in the target sequence after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_part_length", default={'type_scope': 10, 'type':10, 'scope':10, 'subject': 40, 'type_scope_subject': 50},
                        help="The maximum length of (<type>, <scope>) or (<subject>) in the target sequence after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--type_as_special_token", action='store_true',
                        help="set the value of list `unknwon_part_start_limited_choice_list` as special token in tokenizer")
    parser.add_argument("--filter_none_scope", action='store_true',
                        help="filter_none_scope")
    parser.add_argument("--target_is_knwon_part", action='store_true',
                        help="target_is_knwon_part")
    parser.add_argument("--no_pos_no_segment", action='store_true',
                        help="no_pos_no_segment")
    parser.add_argument("--co_teaching", action='store_true',
                        help="co_teaching")
    parser.add_argument("--co_training", action='store_true',
                        help="co_training")
    parser.add_argument("--co_training_method", default=None, type=int,
                        help="co_training_method")
    parser.add_argument("--multi_task", action='store_true',
                        help="multi_task")
    parser.add_argument("--valid_data_not_sample", action='store_true',
                        help="valid_data_not_sample")
    parser.add_argument("--add_prompt", default=None, type=ast.literal_eval,
                        help="Example:{\"type_scope\":<type_scope>, \"subject\":<subject>}; {\"type\":<type>, \"scope\":<scope>, \"subject\":<subject>};")
    parser.add_argument("--select_train_idx_list_file", nargs='+',
                        help="select_train_idx_list_file")
    parser.add_argument("--select_valid_idx_list_file", nargs='+',
                        help="select_valid_idx_list_file")
    parser.add_argument("--select_test_idx_list_file", nargs='+',
                        help="select_test_idx_list_file")
    parser.add_argument("--setting_only_used_to_train", action='store_true',
                        help="setting_only_used_to_train")
    parser.add_argument("--get_training_loss_list", action='store_true',
                        help="get_training_loss_list")
    parser.add_argument("--first_token", default=None,
                        help="first_token")
    parser.add_argument("--eval_bleu_part", default=None, type=ast.literal_eval)
    parser.add_argument("--get_idx_at_last", action='store_true',
                        help="get_idx_at_last")
    parser.add_argument("--base_number", default=np.e, type=float,
                        help="base_number")
    parser.add_argument("--wait_confidence", default=3, type=int,
                        help="wait_confidence")
    parser.add_argument("--max_cpu_count", default=128, type=int,
                        help="wait_confidence")
    parser.add_argument("--calculate_after_each_epoch", action='store_true',
                        help="calculate_after_each_epoch")

    
    # REF TODO
    parser.add_argument('--noise_rate', type = float, 
                        help = 'corruption rate, should be less than 1', default = 0.91)
    parser.add_argument('--forget_rate', type = float, 
                        help = 'forget rate', default = None)
    parser.add_argument('--num_gradual', type = int, default = 10, #???
                        help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
    parser.add_argument('--exponent', type = float, default = 1, 
                        help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
    
    ## fixed
    parser.add_argument("--loss_focus_unknown", action='store_true',
                        help="loss_focus_unknown")
    parser.add_argument("--unknwon_part_start_limited_choice_list", default=None, #["chore", "docs", "feat", "fix", "perf", "refactor", "style", "test"],
                        help="`limited_type_list`")
    parser.add_argument("--skip_ref_token", default=None, # TODO default:None
                        help="ignore_special_token")
    
    parser.add_argument("--wandb_username", type=str, required=False, default=None,
                        help="username in wandb.ai")
    parser.add_argument("--wandb_project_name", default=None, type=str, required=False,
                        help="project name showing in wandb")
    parser.add_argument("--wandb_run_name", default="only_type", type=str, required=False,
                        help="experiment running name showing in wandb")
    args = parser.parse_args()
    
    if not args.wandb_project_name:
        args.wandb_project_name = "CodeT5_cmtgen_{}".format(args.sub_task.replace("/","_")) # "/,\,#,?,%,:"

    if args.forget_rate is None:
        args.forget_rate=args.noise_rate

    if args.task in ['summarize']:
        args.lang = args.sub_task
    elif args.task in ['cmt_msg_gen']:
        args.lang = args.sub_task
    elif args.task in ['refine', 'concode', 'clone']:
        args.lang = 'java'
    elif args.task == 'defect':
        args.lang = 'c'
    elif args.task == 'translate':
        args.lang = 'c_sharp' if args.sub_task == 'java-cs' else 'java'
    
    if args.multi_task:
        args.setting = None
        
    if args.add_prompt is not None:
        args.max_target_length += len(args.add_prompt.keys())
        
    if isinstance(args.setting, dict) and args.max_unknown_part_length is None:
        args.max_unknown_part_length = args.max_part_length['_'.join(args.setting['unknown_part'])]

    if args.co_training_method == 0:
        args.co_training = True
        args.get_idx_at_last = True
        args.get_training_loss_list = True

    if args.get_training_loss_list:
        args.get_idx_at_last = True
    return args


def set_dist(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Setup for distributed data parallel
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    cpu_cont = min(multiprocessing.cpu_count(), args.max_cpu_count)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_cont)
    args.device = device
    args.cpu_cont = cpu_cont
    if isinstance(args.setting, dict) and "type" == args.setting["unknown_part"][0]:
        args.unknwon_part_start_limited_choice = ["feat", "fix", "docs", "style", "refactor", "perf", "test", "chore"]
    if args.filter_none_scope:
        args.skip_ref_token = None


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_limited_type_dict(args, tokenizer):
    limited_type_dict=dict()
    if args.unknwon_part_start_limited_choice_list is not None:
        for limited_type in args.unknwon_part_start_limited_choice_list:
            limited_type_dict[limited_type] = tokenizer.encode("{}".format(limited_type))[1:-1]
    else:
        limited_type_dict=None
    return limited_type_dict

def get_skip_ref_token_id(args, tokenizer):
    skip_ref_token_id = None
    if args.skip_ref_token is not None:
        skip_ref_token_id = tokenizer.encode("{}".format(args.skip_ref_token))[1:-1][0]
    return skip_ref_token_id

def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer, tag=None, output_result=False, teacher_forcing=True, get_loss_list=False):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=args.cpu_cont, pin_memory=True)
    # Start evaluating model
    logger.info("  " + "***** Running ppl evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    model.eval()
    eval_loss, batch_num = 0, 0
    output_part_gen = list()
    output_part_ref = list()
    output_unknown_part_gen = list()
    output_unknown_part_ref = list()
    if args.task == 'cmt_msg_gen':
        limited_type_dict = get_limited_type_dict(args, tokenizer)
        skip_ref_token_id = get_skip_ref_token_id(args, tokenizer)

    if get_loss_list:
        loss_list = list()
        commit_index_list = list()
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
        batch = tuple(t.to(args.device) for t in batch)
        if args.task == 'cmt_msg_gen':
            
            if args.co_teaching:
                if args.get_idx_at_last:
                    source_ids, target_ids, position_ids, segment_ids, noise_or_not, commit_index = batch
                else:
                    source_ids, target_ids, position_ids, segment_ids, noise_or_not = batch
            else:
                if args.get_idx_at_last:
                    source_ids, target_ids, position_ids, segment_ids, commit_index = batch
                else:
                    source_ids, target_ids, position_ids, segment_ids = batch
            if args.no_pos_no_segment:
                position_ids = None
                segment_ids = None
        else:
            source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_type == 'roberta':
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   target_ids=target_ids, target_mask=target_mask)
            else:
                if args.task == 'cmt_msg_gen':
                    if args.setting_only_used_to_train:
                        outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask, \
                                position_ids=position_ids, segment_ids=segment_ids, \
                                skip_ref_token_id = skip_ref_token_id, \
                                teacher_forcing=teacher_forcing)
                    else:
                        outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask, \
                                position_ids=position_ids, segment_ids=segment_ids, \
                                unknown_part_length=args.max_unknown_part_length, unknwon_part_start_flag=tokenizer.sep_token_id, \
                                unknwon_part_start_limited_choice_dict=limited_type_dict, \
                                skip_ref_token_id = skip_ref_token_id, \
                                teacher_forcing=teacher_forcing, loss_focus_unknown=args.loss_focus_unknown) ## TODO revise `decoder_attention_mask`?       
                else:
                    outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss
                
                if args.task == 'cmt_msg_gen' and output_result:
                    for idx in range(len(outputs.logits)):
                        ref_token_list = tokenizer.convert_ids_to_tokens(target_ids[idx])
                        unknwon_part_start_idx = ref_token_list.index(tokenizer.eos_token)+1
                        unknown_part_end_idx = ref_token_list[unknwon_part_start_idx:].index(tokenizer.eos_token) + unknwon_part_start_idx
                        if unknown_part_end_idx <= unknwon_part_start_idx:
                            unknwon_part_start_idx = 0
                            unknown_part_end_idx = -1
                        
                        pred_ids = [torch.argmax(each) for each in outputs.logits[idx]]
                        clean_pred_ids = pred_ids[unknwon_part_start_idx:unknown_part_end_idx]
                        ref_ids = target_ids[idx]
                        clean_ref_ids = ref_ids[unknwon_part_start_idx:unknown_part_end_idx]
                        gen_sentence = tokenizer.decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        clean_gen_sentence = tokenizer.decode(clean_pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        ref_sentence = tokenizer.decode(ref_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        clean_ref_sentence = tokenizer.decode(clean_ref_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        output_part_gen.append(gen_sentence)
                        output_part_ref.append(ref_sentence)
                        output_unknown_part_gen.append(clean_gen_sentence)
                        output_unknown_part_ref.append(clean_ref_sentence)
                if args.task == 'cmt_msg_gen' and get_loss_list:
                    lm_logits = outputs.logits
                    loss_fct = CrossEntropyLoss(ignore_index=0, reduction='none')
                    loss_batch = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), target_ids.view(-1)).view(-1, lm_logits.size(1))
                    loss_batch = torch.nanmean(loss_batch.masked_fill(target_ids==tokenizer.pad_token_id, torch.nan), dim=1)
                    loss_list += loss_batch.tolist()
                    if args.get_idx_at_last:
                        commit_index_list += commit_index.tolist()

        eval_loss += loss.item()
        batch_num += 1
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    if args.task == 'cmt_msg_gen' and output_result:
        os.makedirs(args.res_dir, exist_ok=True)
        output_fn = os.path.join(args.res_dir, "eval_ppl_result_{}.output".format(tag))
        clean_output_fn = os.path.join(args.res_dir, "eval_ppl_result_clean_{}.output".format(tag))
        gold_fn = os.path.join(args.res_dir, "eval_ppl_result_{}.gold".format(tag))
        clean_gold_fn = os.path.join(args.res_dir, "eval_ppl_result_clean_{}.gold".format(tag))
        with open(output_fn, "w") as f:
            f.write("\n".join(output_part_gen))
        with open(gold_fn, "w") as f:
            f.write("\n".join(output_part_ref))
        with open(clean_output_fn, "w") as f:
            f.write("\n".join(output_unknown_part_gen))
        with open(clean_gold_fn, "w") as f:
            f.write("\n".join(output_unknown_part_ref))
    if get_loss_list:
        os.makedirs(args.res_dir, exist_ok=True)
        loss_fn = os.path.join(args.res_dir, "eval_ppl_result_{}.loss".format(tag))
        with open(loss_fn, "w") as f:
            f.write("\n".join(map(lambda x: str(x), loss_list)))
        with open(os.path.join(args.res_dir, "commit_index_list_eval_ppl_{}.pickle".format(tag)), "wb") as f:
            pickle.dump(commit_index_list, f)
    return eval_ppl


def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria, target_is_knwon_part=False, get_score_list=False, get_token_list=False):
    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=args.cpu_cont, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0

    if args.task == 'cmt_msg_gen':
        limited_type_dict = get_limited_type_dict(args, tokenizer)
        skip_ref_token_id = get_skip_ref_token_id(args, tokenizer)
        gold_ids = []
        
    first_token_id=None
    if args.first_token is not None:
        first_token_id = tokenizer.encode("{}".format(args.first_token))
        logger.info("First token is set to `{}`, its token_id is {}".format(args.first_token, first_token_id))

    gen_part_start_idx_batch_idx_list = list()
    score_list = list()
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        batch = tuple(t.to(args.device) for t in batch)
        if args.task == 'cmt_msg_gen':
            if args.co_teaching:
                source_ids, target_token_ids, target_position_ids, target_segment_ids, noise_or_not = batch
            else:
                source_ids, target_token_ids, target_position_ids, target_segment_ids = batch
            if args.no_pos_no_segment:
                target_position_ids = None
                target_segment_ids = None
        else:
            source_ids, _ = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        gen_part_start_idx_before_list = list()

        if args.task == 'cmt_msg_gen':
            known_part_token_ids, known_part_position_ids, known_part_segment_ids = None, None, None
            if first_token_id is not None:
                known_part_token_ids = torch.Tensor(first_token_id*batch[1].shape[0]).view(batch[1].shape[0], len(first_token_id))
                known_part_token_ids = known_part_token_ids.to(args.device)
            if not args.setting_only_used_to_train:
                ## Get known_part (<subject>) 
                ## target_token_ids: known_part(e.g.: <subject>) + <eos> +unknown_part(e.g.: <type> <scope>)
                known_part_token_ids = target_token_ids.clone()
                known_part_position_ids = target_position_ids.clone()
                known_part_segment_ids = target_segment_ids.clone()
                
                for idx in range(batch[1].shape[0]):
                    gen_part_start_idx_before = -1
                    for i, token_id in enumerate(known_part_token_ids[idx]):
                        if i > 1 and token_id == tokenizer.pad_token_id: # 0: # tokenizer.pad_token_id:
                            gen_part_start_idx_before = i
                            break
                    gen_part_start_idx_before_list.append(gen_part_start_idx_before)
                if not target_is_knwon_part:
                    for idx in range(batch[1].shape[0]):
                        gen_part_start_idx_before = gen_part_start_idx_before_list[idx]
                        max_known_part_length = args.max_target_length - args.max_unknown_part_length
                        if gen_part_start_idx_before <= max_known_part_length:
                            ### <subject> token_id
                            known_part_token_ids[idx] = torch.cat((target_token_ids[idx][:gen_part_start_idx_before], \
                            torch.zeros(args.max_target_length-gen_part_start_idx_before, dtype=target_token_ids.dtype, device=args.device)), -1)
                            ### <subject> position_id
                            known_part_position_ids[idx] = torch.cat((target_position_ids[idx][:gen_part_start_idx_before], \
                            torch.zeros(args.max_target_length-gen_part_start_idx_before, dtype=target_position_ids.dtype, device=args.device)), -1)
                            ### <subject> segment_id
                            known_part_segment_ids[idx] = torch.cat((target_segment_ids[idx][:gen_part_start_idx_before], \
                            torch.zeros(args.max_target_length-gen_part_start_idx_before, dtype=target_segment_ids.dtype, device=args.device)), -1)
                    known_part_token_ids = known_part_token_ids[:,:max_known_part_length]
                    known_part_position_ids = known_part_position_ids[:,:max_known_part_length]
                    known_part_segment_ids = known_part_segment_ids[:,:max_known_part_length]
            
            gold = list(target_token_ids.cpu().numpy())
            gold_ids.extend(gold)
            del target_token_ids, target_position_ids, target_segment_ids
            
        with torch.no_grad():
            if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)

                top_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                if args.task == 'cmt_msg_gen':
                    return_dict_in_generate = None
                    output_scores = None
                    if get_score_list:
                        return_dict_in_generate = True
                        output_scores = True
                    
                    gen_part_length = None
                    if not args.setting_only_used_to_train:
                        gen_part_length = args.max_unknown_part_length
                    preds = model.generate(source_ids, 
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=args.beam_size,
                                       early_stopping=True,
                                       max_length=args.max_target_length,
                                       known_output_ids=known_part_token_ids,
                                       known_output_position_ids = known_part_position_ids,
                                       known_output_segment_ids = known_part_segment_ids,
                                       unknown_part_length=gen_part_length,
                                       unknwon_part_start_flag=tokenizer.sep_token_id,
                                       unknwon_part_start_limited_choice_dict=limited_type_dict,
                                       skip_ref_token_id = skip_ref_token_id,
                                       eos_token_id_num = 2,
                                       return_dict_in_generate = return_dict_in_generate,
                                       output_scores = output_scores)
                    if get_score_list:
                        preds, score = preds.sequences, preds.sequences_scores
                    if args.setting_only_used_to_train:
                        for idx, pred in enumerate(preds):
                            gen_part_start_idx_before = pred.tolist().index(tokenizer.sep_token_id)
                            gen_part_start_idx_before_list.append(gen_part_start_idx_before)
                        # gen_part_start_idx_before_list = (preds==tokenizer.sep_token_id).nonzero()[:, 1]
                        # gen_part_start_idx_before_list = list(gen_part_start_idx_before_list.cpu().numpy())
                else:
                    preds = model.generate(source_ids,
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=args.beam_size,
                                       early_stopping=args.task == 'summarize',
                                       max_length=args.max_target_length)
                top_preds = preds
            pred_ids.extend(top_preds)
            if get_score_list:
                score_list += list(score.cpu().numpy())
        gen_part_start_idx_batch_idx_list += gen_part_start_idx_before_list

    pred_nls = [tokenizer.convert_ids_to_tokens(preds) for preds in pred_ids]
    if args.task == 'cmt_msg_gen':
        pred_nls_clean, gold_nls_clean = list(), list()
        for idx in range(len(pred_ids)):
            pred_nls_clean.append(tokenizer.decode(pred_ids[idx][gen_part_start_idx_batch_idx_list[idx]:], \
                skip_special_tokens=True, clean_up_tokenization_spaces=False))
            gold_nls_clean.append(tokenizer.decode(gold_ids[idx][gen_part_start_idx_batch_idx_list[idx]:], \
                skip_special_tokens=True, clean_up_tokenization_spaces=False))

    output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
    gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
    src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))
    if get_score_list:
        score_fn = os.path.join(args.res_dir, "test_{}.score".format(criteria))

    if args.task in ['defect']:
        target_dict = {0: 'false', 1: 'true'}
        golds = [target_dict[ex.target] for ex in eval_examples]
        eval_acc = np.mean([int(p == g) for p, g in zip(pred_nls, golds)])
        result = {'em': eval_acc * 100, 'bleu': 0, 'codebleu': 0}

        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                f.write(pred_nl.strip() + '\n')
                f1.write(target_dict[gold.target] + '\n')
                f2.write(gold.source.strip() + '\n')
            logger.info("Save the predictions into %s", output_fn)
    else:
        dev_accs, predictions = [], []
        if args.task == 'cmt_msg_gen':
            output_fn_clean = os.path.join(args.res_dir, "test_{}_clean.output".format(criteria))
            gold_fn_clean = os.path.join(args.res_dir, "test_{}_clean.gold".format(criteria))
            output_fn_idx_clean = os.path.join(args.res_dir, "test_{}_idx_clean.output".format(criteria))
            gold_fn_idx_clean = os.path.join(args.res_dir, "test_{}_idx_clean.gold".format(criteria))
            with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2, \
                    open(output_fn_clean, "w") as out_clean, open(gold_fn_clean, 'w') as gold_clean, \
                    open(output_fn_idx_clean, "w") as out_idx_clean, open(gold_fn_idx_clean, "w") as gold_idx_clean:
                for gold_raw, preds, pred_nl_clean, gold_nl_clean in zip(eval_examples, \
                            pred_ids, pred_nls_clean, gold_nls_clean):
                    dev_accs.append(pred_nl_clean.strip() == gold_nl_clean.strip())
                    predictions.append(f"{gold_raw.idx}\t{pred_nl_clean}")
                    f.write(f"{gold_raw.idx}\t{tokenizer.decode(preds).strip()}\n")
                    f1.write(f"{gold_raw.idx}\t{gold_raw.target.strip()}\n")
                    f2.write(f"{gold_raw.idx}\t{gold_raw.source.strip()}\n")
                    out_idx_clean.write(f"{gold_raw.idx}\t{pred_nl_clean.strip()}\n")
                    gold_idx_clean.write(f"{gold_raw.idx}\t{gold_nl_clean.strip()}\n")
                    out_clean.write(pred_nl_clean.strip() + '\n')
                    gold_clean.write(gold_nl_clean.strip() + '\n')
            if get_score_list:
                with open(score_fn, 'w') as f:
                    f.write("\n".join(map(lambda x: str(x), score_list)))
            if get_token_list:
                pred_ids_pkl_file_path = os.path.join(args.res_dir, "test_{}_pred_ids.pickle".format(criteria))
                with open(pred_ids_pkl_file_path, "wb") as f:
                    pickle.dump(pred_ids, f)
        else:
            with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
                for pred_nl, gold in zip(pred_nls, eval_examples):
                    dev_accs.append(pred_nl.strip() == gold.target.strip())
                    if args.task in ['summarize']:
                        # for smooth-bleu4 evaluation
                        predictions.append(str(gold.idx) + '\t' + pred_nl)
                        f.write(str(gold.idx) + '\t' + pred_nl.strip() + '\n')
                        f1.write(str(gold.idx) + '\t' + gold.target.strip() + '\n')
                        f2.write(str(gold.idx) + '\t' + gold.source.strip() + '\n')
                    else:
                        f.write(pred_nl.strip() + '\n')
                        f1.write(gold.target.strip() + '\n')
                        f2.write(gold.source.strip() + '\n')


        if args.task == 'summarize':
            (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
            bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        elif args.task == 'cmt_msg_gen':
            (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn_idx_clean)
            bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        else:
            bleu = round(_bleu(gold_fn, output_fn), 2)
            if args.task in ['concode', 'translate', 'refine']:
                codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, args.lang)

        result = {'em': np.mean(dev_accs) * 100, 'bleu': bleu}
        if args.task == 'concode':
            result['codebleu'] = codebleu * 100

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def cal_Prob(one_commit_data):
    idx, score, phais, mean, std = one_commit_data
    return idx, scipy.stats.norm(mean, std).pdf(score)*phais


def EM(data, k = 2, max_step = 200, threhold = 0.00001):
    phais = torch.tensor([[1.0/k] for i in range(k)], device=data.device)
    mean = torch.tensor([[i] for i in range(k)], device=data.device)
    std = torch.tensor([[1] for i in range(k)], device=data.device)
    pi_times_2 = torch.tensor(2 * math.pi)
    for i in range(max_step):
        # Qs = e_step(data,phais,mean,std)
        data_k = data.repeat(k).reshape(k, data.shape[0])
        exponent = torch.pow((data_k - mean),2)*(-1/(2*std))
        Qs = (torch.exp(exponent)/torch.sqrt(pi_times_2*std)*phais)
        Qs = Qs / torch.sum(Qs, dim=0, keepdim=True)
        # phais, mean, std= m_step(data,phais,mean,std,Qs)
        gama_j = torch.sum(Qs, dim=1)
        new_phais = (gama_j/data.shape[0]).reshape(k, 1)
        new_mean = (torch.sum(data*Qs, dim=1)/gama_j).reshape(k, 1)
        X_i_mu_j = torch.pow((data_k - mean),2)
        # new_std = (torch.sum((X_i_mu_j*Qs).transpose(0,1), axis=1) / gama_j).reshape(k, 1)
        new_std = (torch.sum(X_i_mu_j*Qs, axis=1) / gama_j).reshape(k, 1)
        if i > 0 and False not in (torch.abs(new_mean - mean) < threhold):
            break
        phais, mean, std = new_phais, new_mean, new_std
    return phais[:,0].tolist(), mean[:,0].tolist(), std[:,0].tolist()


def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    if args.wandb_username:
        if not args.wandb_project_name:
                args.wandb_project_name = "default"
        wandb.init(project=f"{args.wandb_project_name}", name=args.wandb_run_name, entity="bigcode")
        wandb.config.update(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    if args.co_teaching:
        _, model2, _ = build_or_load_gen_model(args)
    if args.task == 'cmt_msg_gen':
        if args.type_as_special_token:
            tokenizer.add_tokens(args.unknwon_part_start_limited_choice_list, special_tokens=True)
        tokenizer.add_tokens(["NNNone"], special_tokens=True) ### TODO special_token "NNNone"
        if args.add_prompt is not None and isinstance(args.add_prompt, dict):
            for value in args.add_prompt.values():
                tokenizer.add_tokens(value, special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        if args.co_teaching:
            model2.resize_token_embeddings(len(tokenizer))
        limited_type_dict = get_limited_type_dict(args, tokenizer)
        skip_ref_token_id = get_skip_ref_token_id(args, tokenizer)
        
    model.to(args.device)
    if args.co_teaching:
        model2.to(args.device)
    if args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
        if args.co_teaching:
            model2 = torch.nn.DataParallel(model2)
    # pool = multiprocessing.Pool(args.cpu_cont)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.co_teaching:
        # define drop rate schedule
        rate_schedule = np.ones(args.num_train_epochs) * args.forget_rate
        rate_schedule[:args.num_gradual] = np.linspace(0, args.forget_rate**args.exponent, args.num_gradual)

    if args.do_train:
        if args.co_training:
            with open(os.path.join(os.path.dirname(args.train_filename), 'train.msg.txt')) as train_msg_f:
                train_total_num = len(train_msg_f.read().split("\n"))
            logger.info("There are at last {} code diff and commit message can be used to train".format(train_total_num))
        select_train_idx_list = None
        if args.select_train_idx_list_file is not None:
            select_train_idx_list = []
            for each_file_path in args.select_train_idx_list_file:
                with open(each_file_path) as f:
                    select_train_idx_list+=[int(line) for line in f.read().split("\n")]
            select_train_idx_list = sorted(list(set(select_train_idx_list)))
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        with multiprocessing.Pool(args.cpu_cont) as pool:
            train_examples, train_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, 'train',
                                        setting=args.setting, filter_none_scope=args.filter_none_scope, is_sample=False,
                                no_pos_no_segment=args.no_pos_no_segment, select_idx_list=select_train_idx_list,
                                get_idx_at_last = args.get_idx_at_last)
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=args.cpu_cont, pin_memory=True)
        if args.get_training_loss_list:
            os.makedirs(args.res_dir, exist_ok=True)
            with open(os.path.join(args.res_dir, "train_sampler.pickle"), "wb") as f:
                pickle.dump(train_sampler, f)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        if args.co_teaching:
            optimizer_grouped_parameters2 = [
                {'params': [p for n, p in model2.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in model2.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer2 = AdamW(optimizer_grouped_parameters2, lr=args.learning_rate, eps=args.adam_epsilon)
            scheduler2 = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        dev_dataset = {}
        global_step, best_bleu_em, best_ppl = 0, -1, 1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6

        if args.co_teaching:
            global_step2, best_bleu_em2, best_ppl2 = 0, -1, 1e6

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            if args.co_training and cur_epoch > 0 and not args.co_training_method == 0:
                with multiprocessing.Pool(args.cpu_cont) as pool:
                    train_examples, train_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, 'train',
                                    setting=args.setting, filter_none_scope=args.filter_none_scope, is_sample=False,
                                no_pos_no_segment=args.no_pos_no_segment, select_idx_list=select_train_idx_list,
                                get_idx_at_last = args.get_idx_at_last)
                train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
                train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                            num_workers=args.cpu_cont, pin_memory=True)

            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            if args.co_teaching:
                tr_loss2 = 0
            model.train()
            if args.co_teaching:
                model2.train()
                pure_ratio_1_list=[]
                pure_ratio_2_list=[]
            if args.get_training_loss_list:
                loss_list = list()
                commit_index_list = list()
                if args.co_training_method == 0:
                    confidence_mul_loss_list = list()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                
                if args.task == 'cmt_msg_gen' and args.co_teaching:
                    if args.multi_task:
                        if args.get_idx_at_last:
                            source_ids, target_ids_gen_TS, position_ids_gen_TS, segment_ids_gen_TS, \
                                        target_ids_gen_subject, position_ids_gen_subject, segment_ids_gen_subject, noise_or_not, commit_index = batch
                        else:
                            source_ids, target_ids_gen_TS, position_ids_gen_TS, segment_ids_gen_TS, \
                                        target_ids_gen_subject, position_ids_gen_subject, segment_ids_gen_subject, noise_or_not = batch
                    else:
                        if args.get_idx_at_last:
                            source_ids, target_ids, position_ids, segment_ids, noise_or_not, commit_index = batch
                        else:
                            source_ids, target_ids, position_ids, segment_ids, noise_or_not = batch
                        if args.no_pos_no_segment:
                            position_ids, segment_ids = None, None
                else:
                    if args.task == 'cmt_msg_gen' and args.multi_task:
                        if args.get_idx_at_last:
                            source_ids, target_ids_gen_TS, position_ids_gen_TS, segment_ids_gen_TS, \
                                        target_ids_gen_subject, position_ids_gen_subject, segment_ids_gen_subject, commit_index = batch
                        else:
                            source_ids, target_ids_gen_TS, position_ids_gen_TS, segment_ids_gen_TS, \
                                        target_ids_gen_subject, position_ids_gen_subject, segment_ids_gen_subject = batch
                    else:
                        if args.get_idx_at_last:
                            source_ids, target_ids, position_ids, segment_ids, commit_index = batch
                        else:
                            source_ids, target_ids, position_ids, segment_ids = batch
                        if args.no_pos_no_segment:
                            position_ids, segment_ids = None, None
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                if args.task == 'cmt_msg_gen' and args.multi_task:
                    target_mask_gen_TS = target_ids_gen_TS.ne(tokenizer.pad_token_id)
                    target_mask_gen_subject = target_ids_gen_subject.ne(tokenizer.pad_token_id)
                else:
                    target_mask = target_ids.ne(tokenizer.pad_token_id)

                if args.model_type == 'roberta':
                    loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                       target_ids=target_ids, target_mask=target_mask)
                else:
                    if args.task == 'cmt_msg_gen':
                        if args.multi_task:
                            source_ids = torch.concat([source_ids, source_ids], dim=0)
                            source_mask = torch.concat([source_mask, source_mask], dim=0)
                            target_ids = torch.concat([target_ids_gen_TS, target_ids_gen_subject], dim=0)
                            target_mask = torch.concat([target_mask_gen_TS, target_mask_gen_subject], dim=0)
                            position_ids = torch.concat([position_ids_gen_TS, position_ids_gen_subject], dim=0)
                            segment_ids = torch.concat([segment_ids_gen_TS, segment_ids_gen_subject], dim=0)
                            if args.no_pos_no_segment:
                                position_ids, segment_ids = None, None
                        outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask, \
                                position_ids=position_ids, segment_ids=segment_ids, \
                                max_part_length=args.max_part_length,
                                unknwon_part_start_flag=tokenizer.sep_token_id, \
                                unknwon_part_start_limited_choice_dict=limited_type_dict, \
                                skip_ref_token_id = skip_ref_token_id, loss_focus_unknown=args.loss_focus_unknown)
                        if args.co_teaching:
                            if args.no_pos_no_segment:
                                position_ids, segment_ids = None, None
                            outputs2 = model2(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask, \
                                position_ids=position_ids, segment_ids=segment_ids, \
                                unknown_part_length=args.max_unknown_part_length, unknwon_part_start_flag=tokenizer.sep_token_id, \
                                unknwon_part_start_limited_choice_dict=limited_type_dict, \
                                skip_ref_token_id = skip_ref_token_id, loss_focus_unknown=args.loss_focus_unknown)
                    else:
                        outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                    loss = outputs.loss

                    if args.get_training_loss_list:
                        lm_logits = outputs.logits
                        loss_fct = CrossEntropyLoss(ignore_index=0, reduction='none')
                        # logger.info("lm_logits.size:{}".format(lm_logits.size))
                        loss_batch = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), target_ids.view(-1)).view(-1, lm_logits.size(1))
                        loss_batch = torch.nanmean(loss_batch.masked_fill(target_ids==tokenizer.pad_token_id, torch.nan), dim=1)
                        loss_list += loss_batch.tolist()
                        if args.get_idx_at_last:
                            commit_index_list += commit_index.tolist()
                        if cur_epoch > args.wait_confidence and args.co_training and args.co_training_method == 0:
                            loss_batch = [loss * confidence_list[commit_index[idx]] for idx, loss in enumerate(loss_batch)]
                            confidence_mul_loss_list += loss_batch
                            loss_batch = torch.stack(loss_batch, dim=0)
                            loss = torch.mean(loss_batch)
                    
                    if args.co_teaching:
                        lm_logits_1 = outputs.logits
                        lm_logits_2 = outputs2.logits
                        loss_fct = CrossEntropyLoss(ignore_index=0, reduction='none')
                        loss_1 = loss_fct(lm_logits_1.view(-1, lm_logits_1.size(-1)), target_ids.view(-1)).view(-1, 50)
                        loss_2 = loss_fct(lm_logits_2.view(-1, lm_logits_2.size(-1)), target_ids.view(-1)).view(-1, 50)
                        loss_1 = torch.nanmean(loss_1.masked_fill(target_ids==tokenizer.pad_token_id, torch.nan), dim=1)
                        loss_2 = torch.nanmean(loss_2.masked_fill(target_ids==tokenizer.pad_token_id, torch.nan), dim=1)
                        wandb.log({"train_loss (before)": outputs.loss}, step=global_step)
                        wandb.log({"train_loss2 (before)": outputs2.loss}, step=global_step)
                        # Ref: https://github.com/bhanML/Co-teaching/blob/7c7fbe23e15e517db76a0882b6d108e4508e09d6/loss.py#L8-L29
                        ind_1_sorted = torch.argsort(loss_1)
                        loss_1_sorted = loss_1[ind_1_sorted]

                        ind_2_sorted = torch.argsort(loss_2)
                        loss_2_sorted = loss_2[ind_2_sorted]

                        remember_rate = 1 - rate_schedule[cur_epoch]
                        num_remember = int(remember_rate * len(loss_1_sorted))

                        pure_ratio_1 = torch.sum(noise_or_not[ind_1_sorted[:num_remember]])/float(num_remember)
                        pure_ratio_2 = torch.sum(noise_or_not[ind_2_sorted[:num_remember]])/float(num_remember)
                        
                        ind_1_update = ind_1_sorted[:num_remember]
                        ind_2_update = ind_2_sorted[:num_remember]

                        # exchange
                        loss_1_update = loss_fct(lm_logits_1[ind_1_update].view(-1, lm_logits_1[ind_1_update].size(-1)), 
                                            target_ids[ind_1_update].view(-1))
                        loss_2_update = loss_fct(lm_logits_2[ind_2_update].view(-1, lm_logits_2[ind_2_update].size(-1)), 
                                            target_ids[ind_2_update].view(-1))

                        loss, loss2 = torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember
                        pure_ratio_1_list.append(100*pure_ratio_1)
                        pure_ratio_2_list.append(100*pure_ratio_2)

                    if args.wandb_username:
                        wandb.log({"train_loss": loss}, step=global_step)
                        if args.co_teaching:
                            wandb.log({"train_loss2": outputs2.loss}, step=global_step)
                            
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.co_teaching:
                        loss2 = loss2.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    if args.co_teaching:
                        loss2 = loss2 / args.gradient_accumulation_steps
                tr_loss += loss.item()
                if args.co_teaching:
                    tr_loss2 += loss2.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()
                if args.co_teaching:
                    loss2.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    if args.co_teaching:
                        optimizer2.step()
                        optimizer2.zero_grad()
                        scheduler2.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))
                    if args.co_teaching:
                        train_loss2 = round(tr_loss2 * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                        mean_pure_ratio1 = torch.sum(torch.stack(pure_ratio_1_list))/len(pure_ratio_1_list)
                        mean_pure_ratio2 = torch.sum(torch.stack(pure_ratio_2_list))/len(pure_ratio_2_list)
                        bar.set_description("[{}] Train loss1 {} loss2 {} Pure Ratio 1 {} Ratio 2 {}".format(cur_epoch,\
                            round(train_loss, 3), round(train_loss2, 3), round(mean_pure_ratio1.item(),3), round(mean_pure_ratio2.item(),3)))
            if args.get_training_loss_list:
                os.makedirs(args.res_dir, exist_ok=True)
                with open(os.path.join(args.res_dir, "loss_list_cur_epoch_{}.pickle".format(cur_epoch)), "wb") as f:
                    pickle.dump(loss_list, f)
                if args.get_idx_at_last:
                    with open(os.path.join(args.res_dir, "commit_index_list_cur_epoch_{}.pickle".format(cur_epoch)), "wb") as f:
                        pickle.dump(commit_index_list, f)
                if args.co_training_method == 0:
                    with open(os.path.join(args.res_dir, "confidence_mul_loss_list_cur_epoch_{}.pickle".format(cur_epoch)), "wb") as f:
                        pickle.dump(confidence_mul_loss_list, f)

            if args.co_training and cur_epoch >= args.wait_confidence:

                if args.calculate_after_each_epoch:
                    if args.setting_only_used_to_train:
                        eval_setting = None
                    else:
                        eval_setting = args.setting
                    with multiprocessing.Pool(args.cpu_cont) as pool:
                        if args.co_training_method == 0:
                            cache_file_name = "train_dev_data"
                        else:
                            cache_file_name = 'train_dev_{}'.format(cur_epoch)
                        eval_examples, eval_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, cache_file_name, 
                                                setting=eval_setting, filter_none_scope=args.filter_none_scope, is_sample=False, 
                                                no_pos_no_segment=args.no_pos_no_segment, get_idx_at_last = args.get_idx_at_last)
                    eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer, output_result=True, 
                                            tag="train_dev_{}".format(cur_epoch), teacher_forcing=False, get_loss_list=True) # TODO teacher_forcing=True?
                    score_file_path = os.path.join(args.res_dir, "eval_ppl_result_train_dev_{}.loss".format(cur_epoch))
                    commit_index_file = os.path.join(args.res_dir, "commit_index_list_eval_ppl_{}.pickle".format(cur_epoch))
                else:
                    score_file_path = os.path.join(args.res_dir, "loss_list_cur_epoch_{}.pickle".format(cur_epoch))
                    commit_index_file = os.path.join(args.res_dir, "commit_index_list_cur_epoch_{}.pickle".format(cur_epoch))


                if score_file_path.split(".")[-1] == "loss":
                    with open(score_file_path) as score_f:
                        score_list = score_f.read().split("\n")
                elif score_file_path.split(".")[-1] == "pickle":
                    with open(score_file_path, "rb") as score_f:
                        score_list = pickle.load(score_f)
                with open(commit_index_file, "rb") as f:
                    commit_index_list = pickle.load(f)
                logger.info("There are {} commit messages are calculated in the probability".format(len(commit_index_list)))
                if set(commit_index_list) != set(range(len(commit_index_list))):
                    logger.info("Some indexes are missing in commit_index_list")
                unsorted_score_list = [float(item) for item in score_list]
                score_list = []
                for i in np.argsort(commit_index_list):
                    score_list.append(unsorted_score_list[i])

                if args.co_training_method == 0:
                    # Get distribution by EM algorithm
                    ratio, avg, std = EM(torch.tensor(score_list, device=args.device, dtype=torch.float64))
                    if avg[0] < avg[1]:
                        clean_idx = 0
                    else:
                        clean_idx = 1
                    # Get Confidence of loss of each commit
                    # NOT IN ORDER
                    logger.info("Calculating the clean_weight ...")
                    with multiprocessing.Pool(min(multiprocessing.cpu_count(), args.max_cpu_count)) as pool:
                        commit_data = [(idx, score, ratio[clean_idx], avg[clean_idx], std[clean_idx]*9) for idx, score in enumerate(score_list)]
                        idx_clean_weight_list = pool.map(cal_Prob, commit_data)
                    logger.info("Calculating the noisy_weight ...")
                    with multiprocessing.Pool(min(multiprocessing.cpu_count(), args.max_cpu_count)) as pool:
                        commit_data = [(idx, score, ratio[1-clean_idx], avg[1-clean_idx], std[1-clean_idx]) for idx, score in enumerate(score_list)]
                        idx_noisy_weight_list = pool.map(cal_Prob, commit_data)
                    # IN ORDER
                    sorted_idx_clean_weight_list = sorted(idx_clean_weight_list, key=lambda x: x[0])
                    sorted_idx_noisy_weight_list = sorted(idx_noisy_weight_list, key=lambda x: x[0])
                    clean_weight_list = [idx_weight[1] for idx_weight in sorted_idx_clean_weight_list]
                    noisy_weight_list = [idx_weight[1] for idx_weight in sorted_idx_noisy_weight_list]

                    # clean_weight_list = [cal_Prob(each_data, ratio[clean_idx], avg[clean_idx], std[clean_idx]*9) for each_data in score_list]
                    # noisy_weight_list = [cal_Prob(each_data, ratio[1-clean_idx], avg[1-clean_idx], std[1-clean_idx]) for each_data in score_list]
                    logger.info("Calculating the confidence ...")
                    confidence_list = []
                    for idx in range(len(score_list)):
                        confidence_list.append((clean_weight_list[idx]+(np.power(args.base_number, -score_list[idx]))*noisy_weight_list[idx])/(clean_weight_list[idx]+noisy_weight_list[idx]))
                    os.makedirs(args.res_dir, exist_ok=True)
                    with open(os.path.join(args.res_dir, "confidence_list_cur_epoch_{}.pickle".format(cur_epoch)), "wb") as f:
                        pickle.dump(confidence_list, f)
                    logger.info("confidence_list_cur_epoch_{}.pickle saved.".format(cur_epoch))
                elif args.co_training_method == 1:
                    sort_index = list(np.argsort(np.array(score_list)))
                    window = int(len(score_list)/10)
                    select_train_idx_list = []
                    for bin_i in range(10):
                        bin_index = sort_index[int(bin_i*window):int(bin_i*window+window)]
                        select_relative_idx = list(np.random.choice(a=window, size=int(window*(1-0.1*bin_i)), replace=False, p=None))
                        select_train_idx_list += [bin_index[relative_idx] for relative_idx in select_relative_idx]
                elif args.co_training_method == 2:
                    score_prob_list = [max(score_list)-score for score in score_list]
                    select_train_idx_list = random.choices(list(range(len(score_list))), weights=score_prob_list, k= len(score_list))
                elif args.co_training_method == 3:
                    sort_index = list(np.argsort(np.array(score_list)))
                    no_select_train_idx_list = list(set(list(range(train_total_num))).difference(set(select_train_idx_list)))
                    select_n = int((train_total_num-train_example_num)/args.num_train_epochs) # args.num_train_epochs can be changed TODO
                    for i in sort_index[:select_n]:
                        select_train_idx_list.append(no_select_train_idx_list[i])
                    logger.info("There are {}({}) candidates and we select {} from them".format(len(sort_index), len(no_select_train_idx_list), select_n))
                
                if not args.co_training_method == 0:
                    select_train_idx_list = sorted(list(set(select_train_idx_list)))
                    select_train_idx_list_file_path = os.path.join(args.res_dir, "select_train_idx_list_{}.json".format(cur_epoch))
                    with open(select_train_idx_list_file_path, "w") as f:
                        json.dump(select_train_idx_list, f)

            if args.do_eval:
                # Eval model with dev dataset
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    if args.setting_only_used_to_train:
                        eval_setting = None
                    else:
                        eval_setting = args.setting
                    select_valid_idx_list = None
                    if args.select_valid_idx_list_file is not None:
                        select_valid_idx_list = []
                        for each_file_path in args.select_valid_idx_list_file:
                            with open(each_file_path) as f:
                                select_valid_idx_list+=[int(line) for line in f.read().split("\n")]
                        select_valid_idx_list = sorted(list(set(select_valid_idx_list)))
                    with multiprocessing.Pool(args.cpu_cont) as pool:
                        eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 
                                        'dev', setting=eval_setting, filter_none_scope=args.filter_none_scope, is_sample=False,
                                no_pos_no_segment=args.no_pos_no_segment, select_idx_list=select_valid_idx_list,
                                get_idx_at_last = args.get_idx_at_last)
                    dev_dataset['dev_loss'] = eval_examples, eval_data

                eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer, output_result=True, tag=cur_epoch, get_loss_list=True)
                if args.co_teaching:
                    eval_ppl2 = eval_ppl_epoch(args, eval_data, eval_examples, model2, tokenizer, output_result=True, tag=cur_epoch)
                if args.wandb_username:
                    wandb.log({'eval_ppl': eval_ppl}, step=global_step)
                    if args.co_teaching:
                        wandb.log({'eval_ppl2': eval_ppl2}, step=global_step)
                result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_ppl': eval_ppl}
                if args.co_teaching:
                    result['eval_ppl2'] = eval_ppl2
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)
                if args.data_num == -1:
                    tb_writer.add_scalar('dev_ppl', eval_ppl, cur_epoch)
                    if args.co_teaching:
                        tb_writer.add_scalar('dev_ppl2', eval_ppl2, cur_epoch)

                # save last checkpoint
                if args.save_last_checkpoints:
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the last model into %s", output_model_file)
                    if args.co_teaching:
                        model_to_save2 = model2.module if hasattr(model2, 'module') else model2
                        output_model_file2 = os.path.join(last_output_dir, "pytorch_model2.bin")
                        torch.save(model_to_save2.state_dict(), output_model_file2)
                        logger.info("Save the last model(2) into %s", output_model_file2)
                
                # save X0th checkpoint
                if cur_epoch%10==0:
                    epoch_output_dir = os.path.join(args.output_dir, 'checkpoint-epoch{}'.format(cur_epoch))
                    if not os.path.exists(epoch_output_dir):
                        os.makedirs(epoch_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(epoch_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the last model into %s", output_model_file)
                    if args.co_teaching:
                        model_to_save2 = model2.module if hasattr(model2, 'module') else model2
                        output_model_file2 = os.path.join(epoch_output_dir, "pytorch_model2.bin")
                        torch.save(model_to_save2.state_dict(), output_model_file2)
                        logger.info("Save the last model(2) into %s", output_model_file2)

                if eval_ppl < best_ppl and ("eval_ppl2" not in vars() or eval_ppl2 < best_ppl2):
                    not_loss_dec_cnt = 0
                    logger.info("  Best ppl:%s", eval_ppl)
                    if args.co_teaching:
                        logger.info("  Best ppl2:%s", eval_ppl2)
                    logger.info("  " + "*" * 20)
                    fa.write("[%d] Best ppl changed into %.4f\n" % (cur_epoch, eval_ppl))
                    if args.co_teaching:
                        fa.write("[%d] Best ppl2 changed into %.4f\n" % (cur_epoch, eval_ppl2))
                    best_ppl = eval_ppl, 
                    if args.co_teaching:
                        best_ppl2 = eval_ppl2
                    if args.wandb_username:
                        wandb.run.summary["eval_best_ppl"] = eval_ppl
                        if args.co_teaching:
                            wandb.run.summary["eval_best_ppl2"] = eval_ppl2

                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.always_save_model:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the best ppl model into %s", output_model_file)
                        if args.co_teaching:
                            model_to_save2 = model2.module if hasattr(model2, 'module') else model2
                            output_model_file2 = os.path.join(output_dir, "pytorch_model2.bin")
                            torch.save(model_to_save2.state_dict(), output_model_file2)
                            logger.info("Save the best ppl model(2) into %s", output_model_file2)
                else:
                    not_loss_dec_cnt += 1
                    logger.info("Ppl does not decrease for %d epochs", not_loss_dec_cnt)
                    if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                        early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                            cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                        logger.info(early_stop_str)
                        fa.write(early_stop_str)
                        break
                # logger.info("***** CUDA.empty_cache() *****")
                # torch.cuda.empty_cache()
                if args.do_eval_bleu:
                    if args.setting_only_used_to_train:
                        eval_setting = None
                    else:
                        eval_setting = args.setting
                    select_valid_idx_list = None
                    if args.select_valid_idx_list_file is not None:
                        select_valid_idx_list = []
                        for each_file_path in args.select_valid_idx_list_file:
                            with open(each_file_path) as f:
                                select_valid_idx_list+=[int(line) for line in f.read().split("\n")]
                        select_valid_idx_list = sorted(list(set(select_valid_idx_list)))
                    with multiprocessing.Pool(args.cpu_cont) as pool:
                        eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev',
                    only_src=True, is_sample=not args.valid_data_not_sample, setting=eval_setting, filter_none_scope=args.filter_none_scope,
                            no_pos_no_segment=args.no_pos_no_segment, select_idx_list=select_valid_idx_list)
                    

                    result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'dev', 'e%d' % cur_epoch)
                    if args.co_teaching:
                        result2 = eval_bleu_epoch(args, eval_data, eval_examples, model2, tokenizer, 'dev', 'e%d' % cur_epoch)
                    dev_bleu, dev_em = result['bleu'], result['em']
                    if args.co_teaching:
                        dev_bleu2, dev_em2 = result2['bleu'], result2['em']
                    if args.wandb_username:
                        wandb.log({'eval_bleu': dev_bleu}, step=global_step)
                        if args.co_teaching:
                            wandb.log({'eval_bleu2': dev_bleu2}, step=global_step)
                    if args.task in ['summarize', 'cmt_msg_gen']:
                        dev_bleu_em = dev_bleu
                        if args.co_teaching:
                            dev_bleu_em2 = dev_bleu2
                    elif args.task in ['defect']:
                        dev_bleu_em = dev_em
                    else:
                        dev_bleu_em = dev_bleu + dev_em
                    if args.data_num == -1:
                        tb_writer.add_scalar('dev_bleu_em', dev_bleu_em, cur_epoch)
                        if args.co_teaching:
                            tb_writer.add_scalar('dev_bleu_em2', dev_bleu_em2, cur_epoch)
                        # tb_writer.add_scalar('dev_em', dev_em, cur_epoch)
                    if dev_bleu_em > best_bleu_em and ("dev_bleu_em2" not in vars() or dev_bleu_em2 > best_bleu_em2):
                        not_bleu_em_inc_cnt = 0
                        logger.info("  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                                    cur_epoch, dev_bleu_em, dev_bleu, dev_em)
                        if args.co_teaching:
                            logger.info("  [%d] Best bleu2+em2: %.2f (bleu2: %.2f, em2: %.2f)",
                                    cur_epoch, dev_bleu_em2, dev_bleu2, dev_em2)
                        logger.info("  " + "*" * 20)
                        best_bleu_em = dev_bleu_em
                        if args.co_teaching:
                            best_bleu_em2 = dev_bleu_em2
                        if args.wandb_username:
                            wandb.run.summary["eval_best_bleu"] = dev_bleu
                            if args.co_teaching:
                                wandb.run.summary["eval_best_bleu2"] = dev_bleu2
                        fa.write("[%d] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                            cur_epoch, best_bleu_em, dev_bleu, dev_em))
                        if args.co_teaching:
                            fa.write("[%d] Best bleu2+em2 changed into %.2f (bleu2: %.2f, em2: %.2f)\n" % (
                                cur_epoch, best_bleu_em2, dev_bleu2, dev_em2))
                        # Save best checkpoint for best bleu
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or args.always_save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best bleu model into %s", output_model_file)
                            if args.co_teaching:
                                model_to_save2 = model2.module if hasattr(model2, 'module') else model2
                                output_model_file2 = os.path.join(output_dir, "pytorch_model2.bin")
                                torch.save(model_to_save2.state_dict(), output_model_file2)
                                logger.info("Save the best bleu model(2) into %s", output_model_file2)
                    else:
                        not_bleu_em_inc_cnt += 1
                        logger.info("Bleu does not increase for %d epochs", not_bleu_em_inc_cnt)
                        fa.write(
                            "[%d] Best bleu+em (%.2f) does not drop changed for %d epochs, cur bleu+em: %.2f (bleu: %.2f, em: %.2f)\n" % (
                                cur_epoch, best_bleu_em, not_bleu_em_inc_cnt, dev_bleu_em, dev_bleu, dev_em))
                        if args.co_teaching:
                            fa.write(
                                "[%d] Best bleu2+em2 (%.2f) does not drop changed for %d epochs, cur bleu2+em2: %.2f (bleu: %.2f, em: %.2f)\n" % (
                                    cur_epoch, best_bleu_em2, not_bleu_em_inc_cnt, dev_bleu_em2, dev_bleu2, dev_em))
                        if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                            stop_early_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                                cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                            logger.info(stop_early_str)
                            fa.write(stop_early_str)
                            break
            # logger.info("***** CUDA.empty_cache() *****")
            # torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        criteria_list = ['best-ppl'] #'last', 
        if args.do_eval_bleu:
            criteria_list = ['best-bleu', 'last'] #'best-ppl', 
        
        for criteria in criteria_list:
            file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            if not os.path.exists(file):
                continue
            logger.info("Reload model from {}".format(file))
            model.load_state_dict(torch.load(file))
            if args.co_teaching:
                file2 = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model2.bin'.format(criteria))
                if not os.path.exists(file2):
                    continue
                logger.info("Reload model2 from {}".format(file2))
                model2.load_state_dict(torch.load(file2))

            if args.setting_only_used_to_train:
                eval_setting = None
            else:
                eval_setting = args.setting
            select_test_idx_list = None
            if args.select_test_idx_list_file is not None:
                select_test_idx_list = []
                for each_file_path in args.select_test_idx_list_file:
                    with open(each_file_path) as f:
                        select_test_idx_list+=[int(line) for line in f.read().split("\n")]
                select_test_idx_list = sorted(list(set(select_test_idx_list)))
            with multiprocessing.Pool(args.cpu_cont) as pool:
                eval_examples, eval_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer,
                                        'test', only_src=True, is_sample=False, setting=eval_setting,
                                        target_is_knwon_part=args.target_is_knwon_part, 
                                no_pos_no_segment=args.no_pos_no_segment, select_idx_list=select_test_idx_list)
            # eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer, output_result=True, tag="test", teacher_forcing=False)
            # logger.info("test_ppl = %s", eval_ppl)
            if args.do_eval_bleu:
                result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'test', criteria,
                                            target_is_knwon_part=args.target_is_knwon_part)
                if args.co_teaching:
                    result2 = eval_bleu_epoch(args, eval_data, eval_examples, model2, tokenizer, 'test', criteria,
                                                target_is_knwon_part=args.target_is_knwon_part)
                test_bleu, test_em = result['bleu'], result['em']
                test_codebleu = result['codebleu'] if 'codebleu' in result else 0
                result_str = "[%s] bleu-4: %.2f, em: %.4f, codebleu: %.4f\n" % (criteria, test_bleu, test_em, test_codebleu)
                if args.co_teaching:
                    test_bleu2, test_em2 = result2['bleu'], result2['em']
                    test_codebleu2 = result2['codebleu'] if 'codebleu' in result2 else 0
                    result_str += "\n[%s] bleu-4(2): %.2f, em(2): %.4f, codebleu(2): %.4f\n" % (criteria, test_bleu2, test_em2, test_codebleu2)
                logger.info(result_str)
                fa.write(result_str)
            if args.res_fn:
                os.makedirs(os.path.dirname(args.res_fn), exist_ok=True)
                with open(args.res_fn, 'a+') as f:
                    f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                    if args.do_eval_bleu:
                        f.write(result_str)
    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()


if __name__ == "__main__":
    main()
