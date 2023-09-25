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
    # runner.add(max_seq_len=300, use_filter_loss_sum=True, seed=[42, 43, 44, 45, 46])
    # runner.add(max_seq_len=300, use_filter_loss_sum=True, use_rel_ner=True, seed=[42, 43, 44, 45, 46])
    # runner.add(max_seq_len=300, use_rel_ner=True, seed=[42, 43, 44, 45, 46])
    # runner.add(max_seq_len=300, lr=0.00005, seed=[42, 43, 44, 45, 46])


    # Add tests
    runner.add_test(
        use_thres_val=True,
        test_opt1=["last", "best"],
        use_thres_threshold=[0.001, 0.0001],
        test_from_ckpt=["output/ouput-2023-09-24_12-40-02-Kirin-512-MaxLen#300-Fsum"],
        test_batch_size=[1, 2, 4, 8, 16, 32],
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
    # use filter
    use_filter_opt1="F1",
    use_filter_opt2="F2",
    use_filter_opt3="F3",
    use_filter_opt4="F4",
    use_filter_opt5="F5",
    use_filter_opt6="F6",
    use_thres_val="ThresV",
    use_thres_threshold="Thres",
    use_filter_loss_sum="Fsum",
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
)

## Block Configuation
block_configuation = [
    "gpu", "tag", "run_id", "seed", "test_from_ckpt", "offline", "use_cache"
]


if __name__ == "__main__":
    run()
