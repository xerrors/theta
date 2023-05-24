from collections import defaultdict
import os
import argparse
import random
import numpy as np

import wandb
import traceback
import yaml
from main import main
from prettytable import PrettyTable

import xerrors
from xerrors import cprint as cp
from xerrors.utils import get_gpu_by_user_input
from xerrors.metrics import confidence_interval

# 根据 run_config 生成所有的组合
run_id = f"RUN_{xerrors.cur_time()}"

# To Test
index = {
    "use_graph_layers": "G",
    "use_two_plm": "PLM2",
    "use_rel": "Rel-",
    "use_ent_pred_rel": "EntFeature-",
    "use_rel_opt1": "F-", # filter, batch
    "use_gold_ent_val": "GoldEnt",
    "use_gold_filter_val": "GoldFilter",
    "use_filter_hard": "Hard",
    "use_thres_train": "ThresT",
    "use_thres_val": "ThresV",
    "max_epochs": "E",
    "use_filter_opt1": "Fopt1-",
    "use_filter_opt2": "Fopt2-",
    "use_filter_label_enhance": "LabelEn",
    "use_ent_hidden_state": "EntHid-", # head, mean, max
    "use_ent_attn": "AttnE",
    "ent_pair_threshold": "T-",
    "use_spert": "SpERT",
    "na_ner_weight": "NaW-",
    "use_ner": "NER-",
    "use_thres_plus": "ThresT+",
    "context_window": "CW-",
    "precision": "P",
    "task_lr": "LR2-",
}

TAG = "Kappa"

# 用于测试的配置 ===================================
run_config_test = dict(
    use_thres_val=True,
    test_from_ckpt=['output/ouput-2023-05-23_17-03-13-Kappa-Rel-lmhead-EntFeature-embed2/config.yaml', 'output/ouput-2023-05-23_19-56-32-Kappa-Rel-mlp-EntFeature-embed2/config.yaml', 'output/ouput-2023-05-23_22-49-56-Kappa-Rel-lmhead-EntFeature-embed2/config.yaml', 'output/ouput-2023-05-24_01-43-51-Kappa-Rel-mlp-EntFeature-embed2/config.yaml', 'output/ouput-2023-05-24_04-33-39-Kappa-Rel-lmhead-EntFeature-embed2/config.yaml', 'output/ouput-2023-05-24_07-22-37-Kappa-Rel-mlp-EntFeature-embed2/config.yaml', 'output/ouput-2023-05-24_10-11-18-Kappa-Rel-lmhead-EntFeature-embed2/config.yaml', 'output/ouput-2023-05-24_13-02-45-Kappa-Rel-mlp-EntFeature-embed2/config.yaml', 'output/ouput-2023-05-24_15-52-59-Kappa-Rel-lmhead-EntFeature-embed2/config.yaml', 'output/ouput-2023-05-24_18-43-18-Kappa-Rel-mlp-EntFeature-embed2/config.yaml'],
    ent_pair_threshold=[0, 0.001, 0.01, 0.03, 0.05, 0.07],
)

# 用于训练的配置 ====================================
run_config_train = dict(
    # task_lr=[5e-3, 5e-5],
    # use_rel=["lmhead", "mlp"],
    use_thres_plus=True,
    use_ent_pred_rel="embed2",
    seed=[100, 200, 300, 400, 500],
)

# Queue ============================================
# use_rel="mlp"


# 额外的配置 ========================================
run_configs = [
    # {
    #     "use_cache": False,
    # }
]


# ==================================================
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
    task_tag_str = cp.magenta(config['tag'], bold=True)
    cur_time_str = xerrors.cur_time('human')
    cp.info("XJOBS", cur_time_str + str("Runing: ") + task_tag_str)
    cp.print_json(config)

    result = {}

    try:
        result = main(True, **config)
        print(config["tag"], "Done!")

    except KeyboardInterrupt:
        cp.error("XJOBS", "KeyboardInterrupt: Interrupted by user!")
        # wandb.finish()

    except Exception as e:
        cp.error("XJOBS", traceback.format_exc())
        cp.error("XJOBS", f"Running Error: {e}, Continue...")
        # wandb.finish()

    return result

def avg_result(result, key_config):
    '''Acg result from multiple runs with same tag'''
    result_list = defaultdict(list)
    for r in result:
        tag = r['tag']
        result_list[tag].append(r)

    avg_result = []
    for tag, r in result_list.items():
        avg_r = {}
        for key, c in key_config.items():
            formatter = c["formatter"]
            if isinstance(r[0].get(key, 'N/A'), (int, float)):
                avg_r[key] = formatter([i[key] for i in r])
            else:
                avg_r[key] = (r[0].get(key) + f"({len(r)})") if r[0].get(key) else '-'

        avg_result.append(avg_r)

    return avg_result


def default_formatter(data):
    mean, h = confidence_interval(data)
    return f"{mean*100:.2f}"

def conf_formatter(data):
    mean, h = confidence_interval(data)
    return f"{mean*100:.2f} ± {h*100:.2f}"

def args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-F", "--fast-dev-run", action="store_true", help="Fast dev run")
    parser.add_argument("-T", "--test-mode", action="store_true", help="Run Test")
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
    run_configs = get_all_combinations(run_config_test)
else:
    combinations = get_all_combinations(run_config_train)
    run_configs = run_configs + combinations


results = []
for config in run_configs:
    config = refine_config(config, args)

    result = exec_main(config)
    cp.print_json(result)
    result["tag"] = config["tag"]
    results.append(result)

print([result.get("final_config") for result in results])


# 打印表格
cur_time = xerrors.cur_time('human')
cp.success("XJOBS", "All Done! " + cur_time)

table = PrettyTable()


def convert_keylist_to_dict(key_lists):
    key_config = {}
    for key in key_lists:
        if isinstance(key, dict):
            key_config[key["name"]] = key
        else:
            key_config[key] = {
                "name": key,
                "formatter": default_formatter
            }

    return key_config

key_lists = ["tag", {"name": "test_f1", "formatter": conf_formatter}, "test_p", "test_r", "ner_f1", "rel_f1", "best_f1"]
key_config = convert_keylist_to_dict(key_lists)

table.field_names = key_config.keys()
for res in avg_result(results, key_config):
    row = []
    for key in key_config.keys():
        value = res.get(key, 'N/A')
        row.append(value)

    table.add_row(row)

table.align["tag"] = "l"
print(table)