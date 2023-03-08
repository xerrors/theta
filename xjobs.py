from pprint import pprint

from main import main
import time
import utils

# 根据 run_config 生成所有的组合
index = "global_batch_size"
job_id = time.strftime("%Y%m%d-%H%M%S")
run_config = {
    "tag": "model-bsz",
    "gpu": "1",
    "global_batch_size": [32, 64, 128]
}

run_configs = [
    {
        "tag": "win-250",
        "gpu": "1",
        "max_seq_len": 250,
        "context_window": 250
    },
    {
        "tag": "win-350",
        "gpu": "1",
        "max_seq_len": 350,
        "context_window": 350
    },
    {
        "tag": "win-450",
        "gpu": "1",
        "max_seq_len": 450,
        "context_window": 450
    }
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

def exec_main(config):
    try:
        return main(True, **config)

    except KeyboardInterrupt:
        print(utils.red("[XJOBS]"), "KeyboardInterrupt: Interrupted by user!")
        return None

    except Exception as e:
        print(utils.red("[XJOBS]"), "Running Error: {}, Continue...".format(e))
        return None


if __name__ == "__main__":

    # 生成所有的组合
    combinations = get_all_combinations(run_config)

    for i, config in enumerate(combinations):
        config["tag"] = f"{config['tag']}-{config.get(index, i)}"
        print(exec_main(config))

    for config in run_configs:
        print(exec_main(config))

