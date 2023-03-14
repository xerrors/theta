import os
import argparse
from main import main
import time
import utils

# 根据 run_config 生成所有的组合
run_id = "RUN_{}".format(time.strftime("%Y%m%d-%H%M%S"))


index = []
run_config = {}
run_configs = [
    {
        "tag": "lambda-no-win-two-sep",
        "use_two_stage": True,
        "use_independent_plm": True,
    },
    {
        "tag": "lambda-no-win-two",
        "use_two_stage": True,
        "use_independent_plm": False,
    }
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

    config["run_id"] = run_id
    config["fast_dev_run"] = args.fast_dev_run
    config["output"] = args.output
    if not config.get("gpt") or config.get("gpt") not in ["0", "1", "2", "3"]:
        config["gpu"] = GPU

    try:
        return main(True, **config)

    except KeyboardInterrupt:
        print(utils.red("[XJOBS]"), "KeyboardInterrupt: Interrupted by user!")
        return None

    except Exception as e:
        print(utils.red("[XJOBS]"), "Running Error: {}, Continue...".format(e))
        return None


GPU = get_gpu_by_user_input()

parser = argparse.ArgumentParser(add_help=False)
# 添加一个 --fast-dev-run 的 argsparser 配置
parser.add_argument("--fast-dev-run", action="store_true", help="Fast dev run")
parser.add_argument("--output", type=str, default="output", help="Output directory")
args, _ = parser.parse_known_args()

# 生成所有的组合
combinations = get_all_combinations(run_config)

for i, config in enumerate(combinations):
    config["tag"] = f"{config['tag']}"
    for key in index:
        if config.get(key):
            config["tag"] += f"-{key}_{config[key]}"

    print(exec_main(config))

for config in run_configs:
    print(exec_main(config))

