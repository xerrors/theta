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

def before_run(lists, **kwargs):

    f_lists = []
    for item in lists:
        if item.get("gpu") is not None:
            if str(item.get("gpu"))  == kwargs.get("gpu_id"):
                f_lists.append(item)
        else:
            f_lists.append(item)

    lists = sorted(f_lists, key=lambda x: x["seed"] if x.get("seed") else 0)
    return lists

def run():
    # Configure Runner
    runner = Runner(
        name=hostname + "F",
        configuation_index="./configuation_dict.yaml",
        block_configuation=block_configuation,
    )

    # Add arguments
    runner.add(dataset_config="datasets/ace2005/ace2005.yaml", seed=[42, 43, 44, 45, 46])
    # runner.add(dataset_config="datasets/ace2005/ace2005.yaml", use_warmup_rel=5, use_rel_focal_loss=True, gpu=0, seed=[46])
    # runner.add(dataset_config="datasets/ace2005/ace2005.yaml", use_warmup_rel=0, use_rel_focal_loss=True, use_pair_sort_fix=[True, False], gpu=0, seed=[42, 43, 44, 45, 46])

    # runner.add(dataset_config="datasets/ace2005/ace2005.yaml", use_warmup_rel=5, use_ner_focal_loss=True, gpu=1, seed=[44, 45, 46])
    # runner.add(dataset_config="datasets/ace2005/ace2005.yaml", use_warmup_rel=5, use_ner_focal_loss=True, use_rel_focal_loss=True, gpu=1, seed=[42, 43, 44, 45, 46])
    # runner.add(dataset_config="datasets/ace2005/ace2005.yaml", use_warmup_rel=5, use_filter_focal_loss="Newsum", gpu=1, seed=[42, 43, 44, 45, 46])
    # runner.add(dataset_config="datasets/ace2005/ace2005.yaml", use_warmup_rel=5, use_filter_focal_loss=True, use_pair_sort_fix=True, gpu=1, seed=[42, 43, 44, 45, 46])

    # Add tests
    runner.add_test(
        test_opt1=["best"],
        use_thres_threshold=[0.4, 0.3, 0.2],
        test_from_ckpt=["output/ouput-2023-10-20_05-52-43-KE-D5-WarmR#5-FocalF","output/ouput-2023-10-20_02-01-01-KE-D5-WarmR#5-FocalF","output/ouput-2023-10-19_22-11-02-KE-D5-WarmR#5-FocalF"],
        test_batch_size=1,
        offline=True)

    runner.run(main, before_run=before_run, start_index=0)

    # 判断是否有 attibute result_json
    if hasattr(runner, "result_json"):
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
