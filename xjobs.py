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
        name=hostname + "-A512",
        configuation_index="./configuation_dict.yaml",
        block_configuation=block_configuation,
    )

    # Add arguments
    runner.add(use_ent_bio_input=True, seed=[42, 43, 44, 45, 46])
    runner.add(use_ner="lmhead", use_ent_bio_input=True, seed=[42, 43, 44, 45, 46])
    runner.add(use_ner="lmhead", use_ent_bio_input=False, seed=[42, 43, 44, 45, 46])
    runner.add(use_rel_loss_sum=50, rel_rate=0.5, rel_lr=[1e-4, 5e-4], rel_mlp_layer_num=[1, 2], seed=[42, 43, 44, 45, 46])
    runner.add(seed=46)

    # Add tests
    runner.add_test(
        use_thres_val=True,
        test_opt1=["last", "best"],
        use_thres_threshold=[0.0005, 0.0001],
        test_from_ckpt=[],
        test_batch_size=1,
        offline=True,
        )

    runner.run(main,
               sort_by_seed=True,
               skip_index="test_f1",
               skip_value=0.66
               )

    import wandb
    os.makedirs("./output/alert", exist_ok=True)
    wandb.init(project="alert", name=hostname, dir="./output/alert")
    wandb.alert(title=f"{hostname}", text=f"Finished\n{str(runner.result_json)}")

## Block Configuation
block_configuation = [
    "gpu", "tag", "run_id", "seed", "test_from_ckpt", "offline", "use_cache"
]


if __name__ == "__main__":
    run()
