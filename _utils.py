import os
import json
import logging
from preprocess_MCMD import check_angular_convention, get_content

logger = logging.getLogger(__name__)

def add_lang_by_task(target_str, task, sub_task):
    if task == 'summarize':
        target_str = '<en> ' + target_str
    elif task == 'cmt_msg_gen':
        target_str = '<en> ' + target_str
    elif task == 'refine':
        target_str = '<java> ' + target_str
    elif task == 'translate':
        if sub_task == 'java-cs':
            target_str = '<c_sharp> ' + target_str
        else:
            target_str = '<java> ' + target_str
    elif task == 'concode':
        target_str = '<java> ' + target_str
    elif task == 'defect':
        target_str = target_str
    return target_str

def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage, setting, filter_none_scope, target_is_knwon_part, no_pos_no_segment = item

    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(args.task, args.sub_task, example.source)
        else:
            source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source

    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    noise_or_not = False
    assert source_ids.count(tokenizer.eos_token_id) == 1
    if stage == 'test' and args.task not in ['cmt_msg_gen']:
        target_ids = []
    else:
        target_str = example.target
        if args.add_lang_ids:
            target_str = add_lang_by_task(example.target, args.task, args.sub_task)
        if args.task in ['defect', 'clone']:
            if target_str == 0:
                target_str = 'false'
            elif target_str == 1:
                target_str = 'true'
            else:
                raise NameError
        target_str = target_str.replace('</s>', '<unk>')
        
        if args.task in ['cmt_msg_gen']:
            if args.co_teaching:
                noise_or_not = bool(example.noise)
            add_label = None
            if args.multi_task:
                add_label = args.add_prompt
            target_ids_dict = dict()
            target_ids_dict['type_scope'] = tokenizer.encode(get_content(target_str, content_type=['type','scope'], add_label=add_label), \
                                        max_length=args.max_part_length['type_scope'], \
                                        padding='max_length', truncation=True)
            target_ids_dict['subject'] = tokenizer.encode(get_content(target_str, content_type=['subject'], add_label=add_label), \
                                        max_length=args.max_part_length['subject'], \
                                        padding='max_length', truncation=True)
            if setting is None:
                if stage == 'train':
                    logger.warning("args.setting should be set.")
                else:
                    setting = {'front_part':['type','scope'], 'back_part': ['subject']}
            else:
                setting =  {'front_part':setting['known_part'], 'back_part': setting['unknown_part']}
            if target_is_knwon_part:
                target_ids_dict["front_part"] = tokenizer.encode(target_str, \
                                            max_length=args.max_target_length-args.max_part_length['_'.join(setting["back_part"])], \
                                            padding='max_length', truncation=True)
            else:
                if '_'.join(setting["front_part"]) in ['type_scope', 'subject'] and '_'.join(setting["back_part"]) in ['type_scope', 'subject']:
                    target_ids_dict["front_part"] = target_ids_dict['_'.join(setting["front_part"])]
                    target_ids_dict["back_part"] = target_ids_dict['_'.join(setting["back_part"])]
                else:
                    target_ids_dict["front_part"] = tokenizer.encode(get_content(target_str, content_type=setting["front_part"], add_label=add_label), \
                                            max_length=args.max_target_length-args.max_unknown_part_length, \
                                            padding='max_length', truncation=True)
                    target_ids_dict["back_part"] = tokenizer.encode(get_content(target_str, content_type=setting["back_part"], add_label=add_label), \
                                            max_length=args.max_unknown_part_length, \
                                            padding='max_length', truncation=True)
        target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
                                      truncation=True)

        if args.task in ['cmt_msg_gen']:
            target_ids = [tokenizer.pad_token_id] * args.max_target_length
            front_part_eos_idx = target_ids_dict["front_part"].index(tokenizer.eos_token_id)
            target_ids[:front_part_eos_idx] = target_ids_dict["front_part"][:front_part_eos_idx]
            target_ids[front_part_eos_idx] = tokenizer.eos_token_id
            if not target_is_knwon_part:
                target_ids[front_part_eos_idx:front_part_eos_idx+args.max_part_length['_'.join(setting["back_part"])]] = target_ids_dict["back_part"]
                target_ids[front_part_eos_idx] = tokenizer.eos_token_id
                assert target_ids.count(tokenizer.eos_token_id) == 2
            
            position_ids, segment_ids = [0]*(len(target_ids)), [0]*(len(target_ids))
            if not no_pos_no_segment:
                if setting["front_part"] == ['subject']:
                    segment_ids[0] = 1
                eos_idx_list = [i for i, v in enumerate(target_ids) if v == tokenizer.eos_token_id]
                position_ids[1:1+eos_idx_list[0]] = list(range(eos_idx_list[0]))
                segment_ids[1:1+eos_idx_list[0]] = [1]*(eos_idx_list[0])
                if not target_is_knwon_part:
                    position_ids[1+eos_idx_list[0]:1+eos_idx_list[1]] = list(range(eos_idx_list[1]-eos_idx_list[0]))
            return InputFeatures(
                example_index,
                source_ids,
                target_ids,
                position_ids,
                segment_ids,
                noise_or_not,
                index=example.abs_idx,
                url=None #example.url
            )
        else:
            assert target_ids.count(tokenizer.eos_token_id) == 1
            
    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        url=None #example.url
    )


def convert_clone_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
        target_str = "{}: {}".format(args.task, example.target)
    else:
        source_str = example.source
        target_str = example.target
    code1 = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    code2 = tokenizer.encode(target_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    source_ids = code1 + code2
    return CloneInputFeatures(example_index, source_ids, example.label, example.url1, example.url2)


def convert_defect_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source
    code = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    return DefectInputFeatures(example_index, code, example.target)


class CloneInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label,
                 url1,
                 url2
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label
        self.url1 = url1
        self.url2 = url2


class DefectInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 position_ids,
                 segment_ids,
                 noise_or_not,
                 index,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.position_ids = position_ids
        self.segment_ids = segment_ids
        self.noise_or_not = noise_or_not
        self.index = index
        self.url = url


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 abs_idx,
                 source,
                 target,
                 noise=0,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.abs_idx = abs_idx
        self.source = source
        self.target = target
        self.noise = noise
        self.url = url
        self.task = task
        self.sub_task = sub_task


class CloneExample(object):
    """A single training/test example."""

    def __init__(self,
                 code1,
                 code2,
                 label,
                 url1,
                 url2
                 ):
        self.source = code1
        self.target = code2
        self.label = label
        self.url1 = url1
        self.url2 = url2


def read_translate_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            src = line1.strip()
            trg = line2.strip()
            examples.append(
                Example(
                    idx=idx,
                    source=src,
                    target=trg,
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_refine_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0

    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_concode_examples(filename, data_num):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_summarize_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples

def read_cmt_msg_examples(filename, data_num, select_idx_list=None):
    """Read examples from filename."""
    msg_file_path = filename.replace(".diff.txt", ".msg.txt")
    noise_file_path = filename.replace(".diff.txt", ".noise.txt")

    examples = []
    
    with open(filename, encoding="utf-8") as diff_f, open(msg_file_path, encoding="utf-8") as msg_f:
        diff = diff_f.readlines()
        msg = msg_f.readlines()
        if os.path.exists(noise_file_path):
            with open(noise_file_path) as noise_f:
                noise_list = noise_f.read().strip().split("\n")
                noise_list = [int(i.strip()) for i in noise_list]
        else:
            noise_list = [0] * len(msg)

        later_idx = 0
        for idx in range(len(msg)):
            if select_idx_list is None or idx in select_idx_list:
                examples.append(
                    Example(
                        idx=later_idx,
                        abs_idx=idx,
                        source=diff[idx],
                        target=msg[idx],
                        noise=noise_list[idx],
                    )
                )
                if later_idx + 1 == data_num:
                    break
                later_idx+=1

    return examples

def read_defect_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)

            code = ' '.join(js['func'].split())
            examples.append(
                Example(
                    idx=js['idx'],
                    source=code,
                    target=js['target']
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_clone_examples(filename, data_num):
    """Read examples from filename."""
    index_filename = filename
    url_to_code = {}
    with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl') as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())
            url_to_code[js['idx']] = code

    data = []
    with open(index_filename) as f:
        idx = 0
        for line in f:
            line = line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            data.append(CloneExample(url_to_code[url1], url_to_code[url2], label, url1, url2))
            idx += 1
            if idx == data_num:
                break
    return data
