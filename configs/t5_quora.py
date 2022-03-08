# general config settings
output_path = "./work_dirs"
print_output_every = 50
num_workers = 4

# tokenizer
tokenizer_name = "t5-small"

# dataset settings
dataset_name = "Quora"
data_root = "data/quora/"
data = dict(
    mask_pad_token=True,
    task_prefix="paraphrase: ",
    max_length=80,
    train=dict(dataset_length=-1),
    val=dict(train_test_split=0.3, dataset_length=2000,),
)

# train settings
train = dict(
    num_epochs=20,
    batch_size=32,
    learning_rate=1e-4,
    weight_decay=1e-2,
    epoch_length=500,
    accumulation_steps=1,
    with_amp=False,
)

# model settings
model = dict(type="HuggingFaceWrapper", name="t5-small", freeze_encoder=False)

# optimizer settings
optimizer = dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001)
