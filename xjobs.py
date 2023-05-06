from collections import defaultdict
import os
import argparse
import random
import numpy as np

import wandb
import yaml
from main import main
from prettytable import PrettyTable

import xerrors
from xerrors import cprint
from xerrors.utils import get_gpu_by_user_input

# 根据 run_config 生成所有的组合
run_id = f"RUN_{xerrors.cur_time()}"

# To Test

index = {
    "use_graph_layers": "G",
    "use_two_plm": "PLM2",
    "use_rel_opt1": "F-", # filter, batch
    "use_gold_ent_val": "GoldEnt",
    "use_gold_filter_val": "GoldFilter",
    "use_filter_hard": "Hard",
    "use_thres_train": "ThresT",
    "use_thres_val": "ThresV",
    "max_epochs": "E",
    "use_filter_opt1": "Fopt1-",
    "use_filter_label_enhance": "LabelEn",
    "use_ent_hidden_state": "EntHid-", # head, mean, max
    "use_ent_attn": "AttnE",
    "ent_pair_threshold": "T-",
    "use_spert": "SpERT",
    "na_ner_weight": "NaW-",
    "use_ner": "NER-",
}

TAG = "Omicron"

# 用于测试的配置
run_config_test = dict(
    use_thres_val=True,
    # use_filter_hard=True,
    test_from_ckpt=[
        "output/ouput-2023-05-01_00-17-42-Omicron-NER-mlp-AttnE/config.yaml",
        "output/ouput-2023-05-01_03-36-55-Omicron-NER-mlp-AttnE/config.yaml",
        "output/ouput-2023-05-01_06-47-54-Omicron-NER-mlp/config.yaml",
        "output/ouput-2023-05-01_10-00-55-Omicron-NER-mlp/config.yaml",
        "output/ouput-2023-05-02_11-43-11-Omicron-ThresT-NER-mlp-AttnE/config.yaml",
        "output/ouput-2023-05-02_14-30-16-Omicron-ThresT-NER-mlp/config.yaml"
    ],
    ent_pair_threshold=[0.01, 0.03, 0.05, 0.07],
)

# 用于训练的配置
run_config_train = dict(
    use_thres_train=[True, False],
    use_ent_attn=[False, True],
    use_ner="mlp",
    seed=[500, 400],
)

# 额外的配置
run_configs = [
    # {
    #     "use_ent_attn": True
    # }
]


def get_all_combinations(run_config):

    if len(run_config) == 0:
        return []

    combinations = [{}]
    for key, value in run_config.items():
        if isinstance(value, list):
            new_combinations = []
            for item in value:
                for combination in combinations:
                    new_combinations.append({**combination, key: item})
            combinations = new_combinations
        else:
            for combination in combinations:
                combination[key] = value

    return combinations


def refine_config(config, args):

    if args.test_mode:
        with open(config['test_from_ckpt'], 'r') as f:
            ckpt_config = yaml.load(f, Loader=yaml.FullLoader)
            config["tag"] = ckpt_config["tag"]

    else:
        config["tag"] = TAG

    for key, value in config.items():
        if key in index:
            if value is True:
                config["tag"] += f"-{index[key]}"
            elif value is False:
                pass
            else:
                config["tag"] += f"-{index[key]}{value}"

    config["run_id"] = run_id

    if args.fast_dev_run:
        config["fast_dev_run"] = args.fast_dev_run

    config["output"] = args.output

    # GPU
    if not config.get("gpu") or config.get("gpu") not in ["0", "1", "2", "3"]:
        config["gpu"] = GPU

    # Random Seed
    if args.seed != -1 and config.get("seed") is None:
        config["seed"] = args.seed

    return config

def exec_main(config):
    # Log
    task_tag_str = cprint.magenta(config['tag'], bold=True)
    cur_time_str = xerrors.cur_time('human')
    cprint.info("XJOBS", cur_time_str + "Runing: " + task_tag_str)
    cprint.print_json(config)

    result = {}

    try:
        result = main(True, **config)
        print(config["tag"], "Done!")

    except KeyboardInterrupt:
        cprint.error("XJOBS", "KeyboardInterrupt: Interrupted by user!")
        # wandb.finish()

    except Exception as e:
        cprint.error("XJOBS", f"Running Error: {e}, Continue...")
        # wandb.finish()

    return result

def avg_result(result, key_lists):
    '''Acg result from multiple runs with same tag'''
    result_list = defaultdict(list)
    for r in result:
        tag = r['tag']
        result_list[tag].append(r)

    avg_result = []
    for tag, r in result_list.items():
        avg_r = {}
        for key in key_lists:
            # 判断是不是数字
            if isinstance(r[0].get(key, 'N/A'), (int, float)):
                avg_r[key] = np.mean([i.get(key, 'N/A') for i in r])
            else:
                avg_r[key] = (r[0].get(key) + f"({len(r)})") if r[0].get(key) else '-'

        avg_result.append(avg_r)

    return avg_result


def args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fast-dev-run", action="store_true", help="Fast dev run")
    parser.add_argument("--test-mode", action="store_true", help="Run Test")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--gpu", type=str, default="not specified")
    args, _ = parser.parse_known_args()
    return args

args = args_parser()

if args.gpu == "not specified":
    GPU = get_gpu_by_user_input()
else:
    GPU = args.gpu

if args.test_mode:
    run_config = run_config_test
else:
    run_config = run_config_train

# 生成所有的组合
combinations = get_all_combinations(run_config)
run_configs = run_configs + combinations

results = []
for config in run_configs:
    config = refine_config(config, args)

    result = exec_main(config)
    cprint.print_json(result)
    result["tag"] = config["tag"]
    results.append(result)


# 打印表格
cur_time = xerrors.cur_time('human')
cprint.success("XJOBS", "All Done!" + cur_time)

table = PrettyTable()

key_lists = ["tag", "test_f1", "test_p", "test_r", "ner_f1", "rel_f1", "best_f1"]

table.field_names = key_lists
for res in avg_result(results, key_lists):
    row = []
    for key in key_lists:
        value = res.get(key, 'N/A')
        if isinstance(value, float):
            value = f"{value*100:.2f}"

        row.append(value)

    table.add_row(row)

table.align["tag"] = "l"
print(table)