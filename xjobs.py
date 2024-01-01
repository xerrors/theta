import os

os.environ['CURL_CA_BUNDLE'] = ''

from main import main as main
from xerrors.runner import Runner
from dotenv import load_dotenv

load_dotenv()


# 获取主机名
hostname = os.getenv("HOSTNAME", "K")

T = []
ALL = [0, 1, 2, 3, 4]
D5 = "datasets/ace2005/ace2005.yaml"
S2023 = [2023, 2024, 2025, 2026, 2027]
S42 = [42, 43, 44, 45, 46]
S0 = [0, 1, 2, 3, 4]


def run():
    # Configure Runner
    runner = Runner(
        name=hostname + "I",  # n means no focal loss a means with rate
        configuration_index="./configuration_dict.yaml",
        block_configuration=block_configuration)

    global T
    T += ["output/training_output/2024-01-01_07-50-55-KI-BS#32-Fspan-Fstgy#1231-Fdrop#0.3-D4-MaxLen#302-CW#300","output/training_output/2024-01-01_05-51-05-KI-BS#32-Fspan-Fstgy#1231-Fdrop#0.3-D4-MaxLen#302-CW#300","output/training_output/2024-01-01_03-51-01-KI-BS#32-Fspan-Fstgy#1231-Fdrop#0.3-D4-MaxLen#302-CW#300","output/training_output/2023-12-31_23-53-00-KI-BS#32-Fspan-Fstgy#1231-Fdrop#0.3-D4-MaxLen#302-CW#300","output/training_output/2023-12-31_22-09-49-KI-BS#32-Fspan-Fstgy#1231-Fdrop#0.3-D4-MaxLen#302-CW#300"]
    t = [0.5, 0.3, 0.1]
    # runner.add(dataset_config=D5,  use_ner_prompt=True, seed=S42)

    # runner.add(dataset_config=D5, use_wider_ent_decode=True, seed=S0)
    # runner.add(dataset_config=D5, ner_rate=[1,2], mean_loss=[True, False], no_na_in_tag_embedding=True, seed=S0)

    d4_conf = {
        "dataset_config": "datasets/ace2004/ace2004.yaml",
        "max_seq_len": 302,
        "context_window": 300,
        "seed": 2023,
        "data_piece": [0, 1, 2, 3, 4]
    }
    # TODO: use_focal_alpha use_wider_ent_decode use_span_mention
    # runner.add(gpu=1, batch_size=32, use_filter_strategy="1231", filter_dropout_rate=[0.1, 0.3], **d4_conf)
    # runner.add(gpu=0, batch_size=32, use_wider_ent_decode=[True, False], **d4_conf)
    runner.add(gpu=0, batch_size=32, use_span_mention=True, use_ner_prompt=True, **d4_conf)

    # Add tests
    runner.add_test(test_opt1=["best"], test_batch_size=[32, 1], use_thres_threshold=t, offline=True, test_from_ckpt=T)

    runner.run(main,
               before_train_hook=before_train_hook,
               start_index=0,
               main_index="test_f1",
               skip_value=0,
               use_top_config=True)

    mail_alert(runner)
















def mail_alert(runner):
    if hasattr(runner, "result_json") and not runner.args.debug:
        import wandb
        os.makedirs("./output/alert", exist_ok=True)
        wandb.init(project="alert", name=hostname, dir="./output/alert")
        wandb.alert(title=f"{hostname}", text=f"Finished\n{str(runner.result_json)}")
        wandb.finish(quiet=True)


def before_train_hook(runner, lists, **kwargs):

    if runner.args.debug:
        lists = [lists[0]]

    f_lists = []
    for item in lists:
        if item.get("gpu") is not None:
            if str(item.get("gpu")) == kwargs.get("gpu_id"):
                f_lists.append(item)
        else:
            f_lists.append(item)

    if len(f_lists) > 0 and f_lists[0].get("data_piece") is not None:
        sort_key = "data_piece"
    else:
        sort_key = "seed"

    lists = sorted(f_lists, key=lambda x: x[sort_key] if x.get(sort_key) else 0)
    return lists


def demo(name, **kwargs):
    import random
    return {
        "test_f1": random.random(),
        "best_epoch": random.randint(1, 10),
        "best_step": random.randint(1, 10),
    }


# Block Configuration
block_configuration = [
    "gpu", "tag", "run_id", "seed", "test_from_ckpt", "offline", "use_cache", "model_config", "data_piece"
]

if __name__ == "__main__":
    run()
