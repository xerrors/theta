import os
os.environ['CURL_CA_BUNDLE'] = '' # 真是闹了鬼了 https://github.com/huggingface/transformers/issues/17611#issuecomment-1323272726
from main import main
from xerrors.runner import Runner

# 读取环境变量 .env
import os
from dotenv import load_dotenv
load_dotenv()

# 获取主机名
hostname = os.getenv("HOSTNAME", "K")

"""
# Log

2023-1006: D rel 长度是 512
"""


def run():
    # Configure Runner
    runner = Runner(
        name=hostname + "E",
        configuation_index="./configuation_dict.yaml",
        block_configuation=block_configuation,
    )

    # Add arguments
    runner.add(dataset_config="datasets/ace2005/ace2005.yaml", seed=[42, 43, 44, 45, 46])
    # runner.add(dataset_config="datasets/ace2005/ace2005.yaml", na_filter_weight=5, seed=[42, 43, 44, 45, 46])

    # Add tests
    runner.add_test(
        test_opt1=["best"],
        use_thres_threshold=[0.0001],
        test_from_ckpt=["output/ouput-2023-10-11_17-09-43-KDw-D5-NA_Warm#5-NA_Filt#5","output/ouput-2023-10-11_13-31-56-KDw-D5-NA_Warm#5-NA_Filt#5","output/ouput-2023-10-11_09-54-47-KDw-D5-NA_Warm#5-NA_Filt#5","output/ouput-2023-10-11_06-17-50-KDw-D5-NA_Warm#5-NA_Filt#5","output/ouput-2023-10-11_02-41-29-KDw-D5-NA_Warm#5-NA_Filt#5"],
        test_batch_size=1,
        offline=True)

    runner.run(main, sort_by_seed=True, start_index=0)

    import wandb
    os.makedirs("./output/alert", exist_ok=True)
    wandb.init(project="alert", name=hostname, dir="./output/alert")
    wandb.alert(title=f"{hostname}", text=f"Finished\n{str(runner.result_json)}")
    wandb.finish(quiet=True)

## Block Configuation
block_configuation = [
    "gpu", "tag", "run_id", "seed", "test_from_ckpt", "offline", "use_cache", "model_config"
]


if __name__ == "__main__":
    run()
