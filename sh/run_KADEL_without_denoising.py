import subprocess
import argparse
import os

def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='select GPU id')
    parser.add_argument("--gpu_id", default=0, help="experiment running name showing in wandb")
    parser.add_argument('--pl', default="javascript", help="programming language")
    parser.add_argument("--idx", default=0, type=int, help="experiment running name showing in wandb")
    parser.add_argument("--max_cpu_count", default=16, type=int, help="max_cpu_count")
    parser.add_argument('--seed', type=int, default=1234 )
    parser.add_argument('--split_time', action='store_true')
    args = parser.parse_args()
    gpu_id = args.gpu_id
    
    setting= {'known_part':[], 'unknown_part': ['type','scope','subject']}
    setting_str = "{}".format(setting)
    
    data_folder = "../data"
    data_folder_name = "cut_diff200_msg50_with_pseudo_label"
    if args.split_time:
        data_folder_name = f"split_time/{data_folder_name}"
    select_idx_list_file = os.path.join(data_folder, f"cmt_msg_gen/{data_folder_name}/{args.pl}/valid.match_template.txt")
    
    sha="KADEL_without_denoising-{}".format(args.idx)
    if args.seed != 1234:
        sha+="_seed{}".format(args.seed)
    save_dir = os.path.join("saved_models", sha, \
            f"cmt_msg_gen/{data_folder_name}/{args.pl}/codet5_small")
    os.makedirs(save_dir, exist_ok=True)
    pre_script_cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python ../run_gen.py --seed {args.seed} --max_cpu_count {args.max_cpu_count}\
                --setting=\"{setting_str}\" \
                --do_train --do_eval --do_eval_bleu --do_test  \
                --setting_only_used_to_train --no_pos_no_segment \
                --task cmt_msg_gen --sub_task {data_folder_name}/{args.pl} --model_type codet5 \
                --data_num -1  --num_train_epochs 30 --warmup_steps 1000 --learning_rate 5e-5 --patience 30 \
                --tokenizer_name=Salesforce/codet5-small  --model_name_or_path=Salesforce/codet5-small \
                --data_dir {data_folder} \
                --cache_path {save_dir}/cache_data \
                --output_dir {save_dir} \
                --summary_dir tensorboard\
                --save_last_checkpoints --always_save_model \
                --res_dir {save_dir}/prediction \
                --train_batch_size 64 --eval_batch_size 64 \
                --select_valid_idx_list_file {select_idx_list_file} \
                --max_source_length 200 --max_target_length 50 --beam_size 10"
    script_cmd = pre_script_cmd
    script_cmd+=" 2>&1 | tee {}/train.log".format(save_dir)
    print(script_cmd)
    os.system(script_cmd)
    print("="*50)
