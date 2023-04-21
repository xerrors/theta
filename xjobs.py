import os
import argparse
from pprint import pprint
import random

import wandb
from main import main
import time
import utils
from prettytable import PrettyTable

# 根据 run_config 生成所有的组合
run_id = "RUN_{}".format(time.strftime("%Y%m%d-%H%M%S"))

# To Test

index = {
    "use_graph_layers": "G",
    "use_two_plm": "Two-",
    "use_rel_opt1": "F-", # filter, batch
    "use_gold_ent_val": "GoldEnt",
    "use_gold_filter_val": "GoldFilter",
    "use_filter_hard": "Hard",
    "use_thres_train": "ThresT",
    "use_thres_val": "ThresV",
    "max_epochs": "E",
    "use_filter_opt1": "Fopt1-",
    "use_filter_label_enhance": "LabelEn",
    "ent_pair_threshold": "T-",
}


# run_config = dict(
#     tag="Eta",
#     test_from_ckpt="output/ouput-2023-04-21_16-26-27-Eta-ThresV-ThresT-LabelEn-T-0.5-Fopt1-concat/config.yaml",
#     ent_pair_threshold=[0.1, 0.2, 0.3, 0.4]
# )

run_config = dict(
    tag="Eta",
    use_thres_val=True,
    use_thres_train=True,
    use_filter_label_enhance=[True, False],
    ent_pair_threshold=0.3,
    use_filter_opt1="concat",
)

run_configs = [

]


def get_gpu_by_user_input():

    try:
        os.system("gpustat")
    except:
        print("WARNING: Try to install gpustat to check GPU status: pip install gpustat")

    gpu = input("\nSelect GPU >>> ")

    assert gpu and int(gpu) in [0, 1, 2, 3], \
        "Can not run scripts on GPU: {}. Stoped!".format(gpu if gpu else "None")
    print("This scripts will use GPU {}".format(gpu))
    return gpu


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


def exec_main(config):
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(utils.green("\n[XJOBS]"), time_str, f"Running: {utils.magenta(config['tag'])}")
    config["run_id"] = run_id
    config["fast_dev_run"] = args.fast_dev_run
    config["output"] = args.output
    if not config.get("gpu") or config.get("gpu") not in ["0", "1", "2", "3"]:
        config["gpu"] = GPU

    result = None

    try:
        result = main(True, **config)
        print(config["tag"], "Done!")

    except KeyboardInterrupt:
        print(utils.red("\n[XJOBS]"), "KeyboardInterrupt: Interrupted by user!")
        wandb.finish()

    except Exception as e:
        print(utils.red("\n[XJOBS]"), "Running Error: {}, Continue...".format(e))
        wandb.finish()

    return result

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--fast-dev-run", action="store_true", help="Fast dev run")
parser.add_argument("--output", type=str, default="output", help="Output directory")
parser.add_argument("--gpu", type=str, default="not specified")
args, _ = parser.parse_known_args()

if args.gpu == "not specified":
    GPU = get_gpu_by_user_input()
else:
    GPU = args.gpu

# 生成所有的组合
combinations = get_all_combinations(run_config)

results = []

for config in run_configs:
    result = exec_main(config)
    if result:
        pprint(result)
        result["tag"] = config["tag"]
        results.append(result)


for i, config in enumerate(combinations):
    config["tag"] = f"{config['tag']}"
    for key, value in config.items():
        if key in index:
            if value is True:
                config["tag"] += f"-{index[key]}"
            elif value is False:
                pass
            else:
                config["tag"] += f"-{index[key]}{value}"

    result = exec_main(config)
    if result:
        pprint(result)
        result["tag"] = config["tag"]
        results.append(result)


# 打印表格
print(utils.green("\n[XJOBS]"), "All Done!")

table = PrettyTable()

key_lists = ["Tag", "test_f1", "test_p", "test_r", "ner_f1", "rel_f1", "best_f1"]

table.field_names = key_lists
for res in results:
    row = []
    for key in key_lists:
        value = res.get(key.lower(), 'N/A')
        if isinstance(value, float):
            value = f"{value*100:.2f}"

        row.append(value)

    table.add_row(row)

table.align["Tag"] = "l"
print(table)