from pprint import pprint

from main import main

# 根据 run_config 生成所有的组合
run_config = {
    "tag": "xjobs-run",
    "gpu": "1",
    "lr": [1e-4, 1e-5],
    "decoder_lr": [1e-3, 1e-4, 1e-5],
}

# 生成所有数组的组合


def get_all_combinations(run_config):
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


# 生成所有的组合
combinations = get_all_combinations(run_config)

for i, config in enumerate(combinations):
    config["tag"] = f"{config['tag']}-{i}"
    pprint(config)
    main(True, **config)
