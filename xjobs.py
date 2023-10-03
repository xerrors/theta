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
        name=hostname + "-B512",
        configuation_index="./configuation_dict.yaml",
        block_configuation=block_configuation,
    )

    # Add arguments
    # runner.add(use_test_for_val=True, seed=[46])
    runner.add(use_test_for_val=True, batch_size=4, seed=[42, 43, 44, 45, 46])

    # Add tests
    runner.add_test(
        use_thres_val=True,
        test_opt1=["best"],
        use_thres_threshold=[0.0005, 0.0001],
        test_from_ckpt=["output/ouput-2023-10-03_06-30-56-Oxen-A512-Ts4V","output/ouput-2023-10-02_23-40-33-Oxen-A512Z-Ts4V","output/ouput-2023-10-02_22-20-06-Oxen-A512Z-Ts4V","output/ouput-2023-10-02_20-59-44-Oxen-A512Z-Ts4V","output/ouput-2023-10-02_19-31-37-Oxen-A512Z-Ts4V"],
        test_batch_size=1,
        offline=True,
        )

    runner.run(main,
               sort_by_seed=True,
               skip_index="test_f1",
               skip_value=0.67)

    import wandb
    os.makedirs("./output/alert", exist_ok=True)
    wandb.init(project="alert", name=hostname, dir="./output/alert")
    wandb.alert(title=f"{hostname}", text=f"Finished\n{str(runner.result_json)}")
    wandb.finish(quiet=True)

## Block Configuation
block_configuation = [
    "gpu", "tag", "run_id", "seed", "test_from_ckpt", "offline", "use_cache"
]


if __name__ == "__main__":
    run()
