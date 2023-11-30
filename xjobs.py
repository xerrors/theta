from concurrent.futures import ALL_COMPLETED
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

T = []
ALL = [0, 1, 2, 3, 4]
D5 = "datasets/ace2005/ace2005.yaml"
D4 = "datasets/ace2004/ace2004.yaml"
S2023 = [2023, 2024, 2025, 2026, 2027]
S42 = [42, 43, 44, 45, 46]
S0 = [0, 1, 2, 3, 4]

def run():
    # Configure Runner
    runner = Runner(
        name=hostname + "I", # n means no focal loss a means with rate
        configuation_index="./configuation_dict.yaml",
        block_configuation=block_configuation,
    )

    global T
    # T += ["output/ouput-2023-11-20_20-28-58-KI-D5-ProjNorm","output/ouput-2023-11-20_17-39-56-KI-D5-ProjNorm","output/ouput-2023-11-20_15-18-29-KI-D5-ProjNorm","output/ouput-2023-11-19_23-04-19-KI-D5-ProjNorm"]
    # T += ["output/ouput-2023-11-20_10-55-35-KI-D5-FHS-ProjNorm","output/ouput-2023-11-20_08-13-54-KI-D5-FHS-ProjNorm","output/ouput-2023-11-20_05-09-06-KI-D5-FHS-ProjNorm","output/ouput-2023-11-19_23-04-31-KI-D5-FHS-ProjNorm"]
    # T += ["output/ouput-2023-11-20_12-55-12-KI-D5","output/ouput-2023-11-20_10-36-24-KI-D5","output/ouput-2023-11-20_07-56-59-KI-D5","output/ouput-2023-11-20_04-56-52-KI-D5","output/ouput-2023-11-20_01-59-42-KI-D5"]
    # T += ["output/ouput-2023-11-20_21-16-15-KI-D5-FHS","output/ouput-2023-11-20_18-33-49-KI-D5-FHS","output/ouput-2023-11-20_15-58-23-KI-D5-FHS","output/ouput-2023-11-20_13-20-14-KI-D5-FHS","output/ouput-2023-11-20_02-05-59-KI-D5-FHS"]
    T += ["output/ouput-2023-11-27_13-23-50-KI-D5-use_filter_label_enhance","output/ouput-2023-11-27_10-02-21-KI-D5-use_filter_label_enhance"]



    thres = [0.2, 0.18, 0.15, 0.12, 0.10]
    # runner.add(dataset_config=D5, gpu=1, use_filter_focal_loss=["sum", False], use_val_same_as_test=True, seed=[42, 43, 44, 45, 46])
    # TODO tag_marker use_filter_label_enhance in ACE04 (precision 32 in ACE04) (NLLA BIO in 04) use_fix_rate

    runner.add(dataset_config=D5, use_fix_rate=[True, False], use_ner_layer_loss="Bio", seed=S0)
    # runner.add(dataset_config=D5, use_rel_loss_sum=100, use_dynamic_loss_sum=True, use_rel_na_warmup=0, seed=S2023)
    # runner.add(dataset_config=D5, use_ner_layer_loss=[0, "Bio", True], seed=S0)
    # runner.add(dataset_config=D5, gpu=1, use_cross_ner=True, use_ner_layer_loss=[0, "Bio"], ner_rate=2, seed=S0)
    # runner.add(dataset_config=D5, gpu=1, use_cross_ner=True, context_window=[100, 200], seed=S0)

    #! D4 1109
    # runner.add(dataset_config=D4, use_fix_rate=[True, False], use_ner_layer_loss="Bio", seed=2023, data_piece=ALL)

    # Add tests
    runner.add_test(test_opt1=["best"], test_batch_size=1, use_thres_threshold=thres, offline =True, test_from_ckpt=T)

    runner.run(main,
               before_run_hook=before_run_hook,
               start_index=0,
               main_index="test_f1",
               skip_value=0.68,
               use_top_config=True)

    mail_alert(runner)
















def mail_alert(runner):
    if hasattr(runner, "result_json"):
        import wandb
        os.makedirs("./output/alert", exist_ok=True)
        wandb.init(project="alert", name=hostname, dir="./output/alert")
        wandb.alert(title=f"{hostname}", text=f"Finished\n{str(runner.result_json)}")
        wandb.finish(quiet=True)

def before_run_hook(lists, **kwargs):

    f_lists = []
    for item in lists:
        if item.get("gpu") is not None:
            if str(item.get("gpu"))  == kwargs.get("gpu_id"):
                f_lists.append(item)
        else:
            f_lists.append(item)

    lists = sorted(f_lists, key=lambda x: x["seed"] if x.get("seed") else 0)
    return lists

def demo(name, **kwargs):
    import random
    return {
        "test_f1": random.random(),
        "best_epoch": random.randint(1, 10),
        "best_step": random.randint(1, 10),
    }


## Block Configuation
block_configuation = [
    "gpu", "tag", "run_id", "seed", "test_from_ckpt", "offline", "use_cache", "model_config", "data_piece"
]

if __name__ == "__main__":
    run()
