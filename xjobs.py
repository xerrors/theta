import os
import argparse
from main import main
import time
import utils

# 根据 run_config 生成所有的组合
run_id = "RUN_{}".format(time.strftime("%Y%m%d-%H%M%S"))


index = ["use_ner"]
run_config = dict(
    tag="zeta",
    use_ner="multi_classifier",
    use_rel_cls="multi_classifier",
    use_entity_pair_filter=["bilinear", "cat_and_cls", "proj_then_cat", "attention"],
)
run_configs = []

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

    config["run_id"] = run_id
    config["fast_dev_run"] = args.fast_dev_run
    config["output"] = args.output
    if not config.get("gpu") or config.get("gpu") not in ["0", "1", "2", "3"]:
        config["gpu"] = GPU

    try:
        return main(True, **config)

    except KeyboardInterrupt:
        print(utils.red("[XJOBS]"), "KeyboardInterrupt: Interrupted by user!")
        return None

    except Exception as e:
        print(utils.red("[XJOBS]"), "Running Error: {}, Continue...".format(e))
        return None



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

for config in run_configs:
    print(exec_main(config))

for i, config in enumerate(combinations):
    config["tag"] = f"{config['tag']}"
    for key in index:
        if config.get(key):
            config["tag"] += f"-{key}_{config[key]}"

    print(exec_main(config))

