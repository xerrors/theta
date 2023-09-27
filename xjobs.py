from main import main
from xerrors.runner import Runner

def run():
    # Configure Runner
    runner = Runner(
        name="Kirin-512",
        configuation_index=configuation_index,
        block_configuation=block_configuation,
    )

    # Add arguments
    runner.add(use_ent_bio_input=True, seed=[42, 43, 44, 45, 46])
    runner.add(use_ner="lmhead", use_ent_bio_input=True, seed=[42, 43, 44, 45, 46])
    runner.add(use_ner="lmhead", use_ent_bio_input=False, seed=[42, 43, 44, 45, 46])
    runner.add(use_rel_loss_sum=50, rel_rate=0.5, rel_lr=[1e-4, 5e-4], rel_mlp_layer_num=[1, 2], seed=[42, 43, 44, 45, 46])
    # gamma
    runner.add(use_thres_gamma=[0.1, 0.3, 0.5, 0.7], use_filter_strategy=["0927", "0928"], seed=[42, 43, 44, 45, 46])
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
configuation_index = dict(
    # Basic
    max_epochs="E",
    lr="LR",
    batch_size="BS",
    test_batch_size="TBS",
    max_seq_len="MaxLen",
    optimizer="Opt",
    # use ner
    use_ner="NER",
    ent_attn_layer_num="AttnL",
    ent_mlp_layer_num="MLPL",
    # use rel
    use_rel="REL",
    use_rel_opt1="R1",
    use_rel_opt2="R2",
    use_rel_opt3="R3",
    use_rel_opt4="R4",
    use_rel_ner="RN",
    use_rel_loss_sum="Rsum",
    rel_mlp_layer_num="RelMLP",
    # use filter
    use_filter_opt1="F1",
    use_filter_opt2="F2",
    use_filter_opt4="F4",
    use_filter_opt5="F5",
    use_filter_opt6="F6",
    use_thres_val="ThresV",
    use_thres_threshold="Thres",
    use_filter_loss_sum="Fsum",
    use_filter_strategy="Fstrtegy",
    # Task
    context_window="CW",
    # Test
    test_opt1="T1",
    test_opt2="T2",
    test_opt3="T3",
    test_opt4="T4",
    # rate
    filter_rate="FR",
    rel_rate="RR",
    ner_rate="NR",
    rel_ner_rate="RNR",
)

## Block Configuation
block_configuation = [
    "gpu", "tag", "run_id", "seed", "test_from_ckpt", "offline", "use_cache"
]


if __name__ == "__main__":
    run()
