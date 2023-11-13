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
        name=hostname + "H",
        configuation_index="./configuation_dict.yaml",
        block_configuation=block_configuation,
    )

    global T
    # T += ["output/ouput-2023-11-11_15-28-19-KH-D5-LenEmb","output/ouput-2023-11-11_12-24-41-KH-D5-LenEmb","output/ouput-2023-11-11_09-21-18-KH-D5-LenEmb","output/ouput-2023-11-11_06-18-43-KH-D5-LenEmb","output/ouput-2023-11-11_00-27-18-KH-D5-LenEmb"]
    # T += ["output/ouput-2023-11-11_12-50-30-KH-D4","output/ouput-2023-11-11_10-48-20-KH-D4","output/ouput-2023-11-11_08-45-22-KH-D4","output/ouput-2023-11-11_04-32-14-KH-D4","output/ouput-2023-11-11_02-31-05-KH-D4"]
    # T += ["output/ouput-2023-11-11_19-14-17-KH-D4-LenEmb","output/ouput-2023-11-11_17-03-53-KH-D4-LenEmb","output/ouput-2023-11-11_14-52-11-KH-D4-LenEmb","output/ouput-2023-11-11_06-34-48-KH-D4-LenEmb","output/ouput-2023-11-11_00-24-13-KH-D4-LenEmb"]
    # T += ["output/ouput-2023-11-11_21-25-32-KH-D5","output/ouput-2023-11-11_18-32-43-KH-D5","output/ouput-2023-11-11_03-26-40-KH-D5","output/ouput-2023-11-10_20-22-59-KH-D5","output/ouput-2023-11-10_17-33-58-KH-D5"]
    # T += ["output/ouput-2023-11-12_09-10-28-KH-D4-LenEmb-use_rel_refine","output/ouput-2023-11-12_07-00-12-KH-D4-LenEmb-use_rel_refine","output/ouput-2023-11-12_04-49-19-KH-D4-LenEmb-use_rel_refine","output/ouput-2023-11-12_02-39-02-KH-D4-LenEmb-use_rel_refine","output/ouput-2023-11-12_00-32-47-KH-D4-LenEmb-use_rel_refine"]
    T += ["output/ouput-2023-11-13_06-03-01-KH-D5-LenEmb-Fstgy#1109-Neg#init2","output/ouput-2023-11-13_03-11-43-KH-D5-LenEmb-Fstgy#1109-Neg#init2","output/ouput-2023-11-13_00-16-53-KH-D5-LenEmb-Fstgy#1109-Neg#init2","output/ouput-2023-11-12_21-20-22-KH-D5-LenEmb-Fstgy#1109-Neg#init2","output/ouput-2023-11-12_03-31-54-KH-D5-LenEmb-Fstgy#1109-Neg#init2"]
    # T += ["output/ouput-2023-11-12_18-26-33-KH-D5-LenEmb-use_rel_refine","output/ouput-2023-11-12_15-24-16-KH-D5-LenEmb-use_rel_refine","output/ouput-2023-11-12_12-32-22-KH-D5-LenEmb-use_rel_refine","output/ouput-2023-11-12_09-30-23-KH-D5-LenEmb-use_rel_refine","output/ouput-2023-11-12_00-32-29-KH-D5-LenEmb-use_rel_refine"]



    thres = [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
    thres_focal = [0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    # runner.add(dataset_config=D5, gpu=1, use_filter_focal_loss=["sum", False], use_val_same_as_test=True, seed=[42, 43, 44, 45, 46])
    # TODO use_rel_refine use_filter_strategy 1109 random2 init2

    runner.add(dataset_config=D5, gpu=0, use_cache=False, seed=[2, 3, 4])
    # runner.add(dataset_config=D5, use_length_embedding=True, gpu=0, use_rel_refine=True, seed=S2023)
    # runner.add(dataset_config=D5, use_length_embedding=True, gpu=0, use_negative="init2", use_rel_refine=True, seed=S2023)
    # runner.add(dataset_config=D4, use_length_embedding=True, gpu=1, use_rel_refine=True, seed=2023, data_piece=ALL)
    # runner.add(dataset_config=D5, gpu=1, use_filter_focal_loss="sum", use_negative=["noise2", "random2"], seed=S42)
    # Add tests
    runner.add_test(test_opt1=["best"], test_batch_size=1, use_thres_threshold=thres_focal, offline=True, test_from_ckpt=T)

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
