from main import main
from xerrors.runner import Runner

# 读取环境变量 .env
import os
from dotenv import load_dotenv
load_dotenv()

# 获取主机名
hostname = os.getenv("HOSTNAME", "Kirin")

def run():
    # Configure Runner
    runner = Runner(
        name=hostname + "-C512",
        configuation_index="./configuation_dict.yaml",
        block_configuation=block_configuation,
    )

    # Add arguments
    runner.add(dataset_config="datasets/ace2004/ace2004.yaml", use_filter_shuffle=True,  seed=[42, 43, 44, 45, 46])
    runner.add(dataset_config="datasets/ace2005/ace2005.yaml", use_filter_shuffle=True,  seed=[42, 43, 44, 45, 46])
    runner.add(dataset_config="datasets/ace2005/ace2005.yaml", use_filter_shuffle=True, use_mask=True, seed=[42, 43, 44, 45, 46])

    # Add tests
    runner.add_test(
        use_thres_val=True,
        test_opt1=["best"],
        use_thres_threshold=[0.0005, 0.0001, 0.00005],
        test_from_ckpt=[],
        test_batch_size=1,
        offline=True)

    runner.run(main, sort_by_seed=True)

    import wandb
    os.makedirs("./output/alert", exist_ok=True)
    wandb.init(project="alert", name=hostname, dir="./output/alert")
    wandb.alert(title=f"{hostname}", text=f"Finished\n{str(runner.result_json)}")
    wandb.finish(quiet=True)

## Block Configuation
block_configuation = [
    "gpu", "tag", "run_id", "seed", "test_from_ckpt", "offline", "use_cache", "dataset_config"
]


if __name__ == "__main__":
    run()
