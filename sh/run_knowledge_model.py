import subprocess
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='select GPU id')
    parser.add_argument("--gpu_id", default=0, help="experiment running name showing in wandb")
    parser.add_argument("--idx", default=0, type=int, help="experiment running name showing in wandb")
    parser.add_argument("-pl", "--prgramming_language", default="javascript")
    parser.add_argument("-kl", "--knowledge", default="scope", choices=["type", "scope"])
    parser.add_argument("--max_cpu_count", default=16, type=int, help="max_cpu_count")
    parser.add_argument("--seed", type=int, default=1234 )
    parser.add_argument("--data_dir_reverse", action='store_true')
    parser.add_argument("--type_as_special_token", action='store_true')
    parser.add_argument("--finetuned_js", action='store_true')

    args = parser.parse_args()
    gpu_id = args.gpu_id
    
    setting= {'known_part':['subject'], 'unknown_part': [args.knowledge]}
    setting_str = "{}".format(setting)
    
    data_folder = "../data"
    data_folder_name = "sort_time/cut_diff200_msg50" ## TODO
    sub_task = f"{data_folder_name}/{args.prgramming_language}"

    if args.data_dir_reverse:
        sub_task = f"{args.prgramming_language}/{data_folder_name}"

    match_template_file_dict = dict()
    for split_type in ["train", "valid", "test"]:
        match_template_file_dict[split_type] = os.path.join(data_folder, "cmt_msg_gen", sub_task, f"{split_type}.match_template.txt")
    
    sha="KADEL_knowledge-{}-{}".format(args.knowledge, args.idx)
    if args.seed != 1234:
        sha+="_seed{}".format(args.seed)
    save_dir = os.path.join("saved_models", sha, "cmt_msg_gen/{}/{}/codet5_small".format(\
                                        args.prgramming_language, data_folder_name))
    os.makedirs(save_dir, exist_ok=True)
    # --select_test_idx_list_file {match_template_file_dict['test']} \
    pre_script_cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python ../run_gen.py \
                --seed {args.seed} --max_cpu_count {args.max_cpu_count} \
                --setting=\"{setting_str}\" \
                --select_train_idx_list_file {match_template_file_dict['train']} \
                --select_valid_idx_list_file {match_template_file_dict['valid']} \
                --do_train --do_eval --do_eval_bleu --do_test  \
                --task cmt_msg_gen --sub_task {sub_task} --model_type codet5 \
                --data_num -1  --num_train_epochs 30 --warmup_steps 1000 \
                --learning_rate 5e-5 --patience 2 \
                --tokenizer_name=Salesforce/codet5-small \
                --model_name_or_path=Salesforce/codet5-small \
                --data_dir {data_folder} \
                --cache_path {save_dir}/cache_data \
                --output_dir {save_dir} \
                --summary_dir tensorboard\
                --save_last_checkpoints --always_save_model \
                --res_dir {save_dir}/prediction --max_unknown_part_length 10 \
                --train_batch_size 64 --eval_batch_size 64 \
                --max_source_length 200 --max_target_length 50 --beam_size 10"
    if args.knowledge == "scope":
        script_cmd = pre_script_cmd + " --filter_none_scope"
    else:
        script_cmd = pre_script_cmd

    if args.type_as_special_token:
        script_cmd += " --type_as_special_token"
    if args.finetuned_js:
        script_cmd += f" --finetuned_js {args.knowledge}"

    script_cmd+=" 2>&1 | tee {}/train.log".format(save_dir)
    print(script_cmd)
    os.system(script_cmd)
    print("="*50)




'''
TYPE:   subject-to-type-filter_none_scope_False-None2NNNone_True
SCOPE:  subject-to-scope-filter_none_scope_True-None2NNNone_False
save_dir = saved_models/657800c_fdu_170_1/subject-to-scope-filter_none_scope_True-None2NNNone_False/\
cmt_msg_gen/javascript/raw/codet5_small

Namespace(adam_epsilon=1e-08, add_lang_ids=False, add_task_prefix=False, always_save_model=True, 
beam_size=10, 
cache_path='{save_dir}/cache_data', 
co_teaching=False, config_name='', data_dir='../data', data_num=-1, dev_filename=None, 
do_eval=True, do_eval_bleu=True, do_lower_case=False, do_test=True, do_train=True, 
eval_batch_size=64, eval_steps=-1, eval_task='', exponent=1, filter_none_scope=True, 
forget_rate=0.91, gradient_accumulation_steps=1, lang='javascript/raw', learning_rate=5e-05, 
load_model_path=None, local_rank=-1, log_steps=-1, loss_focus_unknown=False, max_grad_norm=1.0, 
max_source_length=200, max_steps=-1, max_target_length=50, max_unknown_part_length=10, 
model_name_or_path='Salesforce/codet5-small', model_type='codet5', no_cuda=False, 
noise_rate=0.91, num_gradual=10, num_train_epochs=30, 
output_dir='{save_dir}', 
patience=2, 
res_dir='{save_dir}/prediction', 
res_fn=None, save_last_checkpoints=True, save_steps=-1, seed=1234, 
setting={'known_part': ['subject'], 'unknown_part': ['scope']}, 
skip_ref_token=None, start_epoch=0, sub_task='javascript/raw', summary_dir='tensorboard', 
target_is_knwon_part=False, task='cmt_msg_gen', test_filename=None, 
tokenizer_name='Salesforce/codet5-small', train_batch_size=64, train_filename=None, 
train_steps=-1, type_as_special_token=False, unknwon_part_start_limited_choice_list=None, 
wandb_project_name='CodeT5_cmtgen_javascript_raw', wandb_run_name='only_type', 
wandb_username='weit', warmup_steps=1000, weight_decay=0.0)
'''
