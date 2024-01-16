import subprocess
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='select GPU id')
    parser.add_argument("--gpu_id", default=0, help="experiment running name showing in wandb")
    parser.add_argument("--idx", default=0, type=int, help="experiment running name showing in wandb")
    parser.add_argument("--max_cpu_count", default=16, type=int, help="max_cpu_count")
    parser.add_argument('--seed', type=int, default=1234 )
    args = parser.parse_args()
    gpu_id = args.gpu_id
    
    setting= {'known_part':[], 'unknown_part': ['subject']}
    setting_str = "{}".format(setting)
    
    data_folder = "../data"
    data_folder_name = "cut_diff200_msg50_with_pseudo_label"
    
    sha="KADEL_without_knowledge-{}".format(args.idx)
    if args.seed != 1234:
        sha+="_seed{}".format(args.seed)
    save_dir = os.path.join("saved_models", sha, "cmt_msg_gen/javascript/{}/codet5_small".format(data_folder_name))
    os.makedirs(save_dir, exist_ok=True)
    pre_script_cmd = "CUDA_VISIBLE_DEVICES={} python ../run_gen.py --seed {} --max_cpu_count {}\
                --setting=\"{}\" \
                --do_train --do_eval --do_eval_bleu --do_test  \
                --setting_only_used_to_train --no_pos_no_segment \
                --task cmt_msg_gen --sub_task {}/javascript --model_type codet5 \
                --data_num -1  --num_train_epochs 30 --warmup_steps 1000 --learning_rate 5e-5 --patience 30 \
                --tokenizer_name=Salesforce/codet5-small  --model_name_or_path=Salesforce/codet5-small \
                --data_dir {} \
                --cache_path {}/cache_data \
                --output_dir {} \
                --summary_dir tensorboard\
                --save_last_checkpoints --always_save_model \
                --res_dir {}/prediction \
                --train_batch_size 64 --eval_batch_size 64 \
                --max_source_length 200 --max_target_length 50 --beam_size 10".format(gpu_id, args.seed, args.max_cpu_count,
                setting_str, data_folder_name, data_folder, save_dir, save_dir, save_dir)
    script_cmd = pre_script_cmd
    script_cmd+=" 2>&1 | tee {}/train.log".format(save_dir)
    print(script_cmd)
    os.system(script_cmd)
    print("="*50)
