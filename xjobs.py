from main import main
from xerrors.runner import Runner

def run():
    # Configure Runner
    runner = Runner(
        name="Oxen-512",
        configuation_index="./configuation_dict.yaml",
        block_configuation=block_configuation,
    )

    # Add arguments
    runner.add(use_ent_bio_input=True, seed=[42, 43, 44, 45, 46])
    runner.add(use_ner="lmhead", use_ent_bio_input=True, seed=[42, 43, 44, 45, 46])
    runner.add(use_ner="lmhead", use_ent_bio_input=False, seed=[42, 43, 44, 45, 46])
    runner.add(use_rel_loss_sum=50, rel_rate=0.5, rel_lr=[1e-4, 5e-4], rel_mlp_layer_num=[1, 2], seed=[42, 43, 44, 45, 46])
    # gamma
    runner.add(use_thres_gamma=0.5, use_filter_strategy=["0511"], seed=[42, 43, 44, 45, 46])
    runner.add(seed=46)

    # Add tests
    runner.add_test(
        use_thres_val=True,
        test_opt1=["last", "best"],
        use_thres_threshold=[0.0005, 0.0001],
        test_from_ckpt=["output/ouput-2023-09-25_09-29-42-Kirin-512-MaxLen#300-Fsum-RN","output/ouput-2023-09-25_04-41-54-Kirin-512-MaxLen#300-Fsum-RN","output/ouput-2023-09-24_23-53-51-Kirin-512-MaxLen#300-Fsum-RN","output/ouput-2023-09-24_19-05-17-Kirin-512-MaxLen#300-Fsum-RN","output/ouput-2023-09-24_14-13-11-Kirin-512-MaxLen#300-Fsum-RN"],
        test_batch_size=1,
        offline=True,
        )

    runner.run(main,
               sort_by_seed=True,
               skip_index="test_f1",
               skip_value=0.66
               )

## Configuation Index

## Block Configuation
block_configuation = [
    "gpu", "tag", "run_id", "seed", "test_from_ckpt", "offline", "use_cache"
]


if __name__ == "__main__":
    run()
