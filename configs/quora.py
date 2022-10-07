# dataset settings
dataset_name = "Quora"
data_root = "data/quora/"
data = dict(
    max_length=80,
    train=dict(dataset_length=-1),
    val=dict(
        train_test_split=0.3,
        dataset_length=2000,
    ),
)
