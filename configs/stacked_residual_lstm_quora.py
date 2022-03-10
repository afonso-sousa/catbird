# general config settings
output_path = "./work_dirs"
print_output_every = 50
num_workers = 4

# dataset settings
dataset_name = "Quora"
data_root = "data/quora/"
data = dict(
    max_length=80,
    train=dict(dataset_length=-1),
    val=dict(train_test_split=0.3, dataset_length=2000,),
)

# train settings
train = dict(
    num_epochs=10,
    batch_size=32,
    accumulation_steps=1,
    with_amp=False,
)

# model settings
model = dict(
    type="StackedResidualLSTM",
    encoder=dict(type="RecurrentEncoder", num_layers=2,),
    decoder=dict(type="RecurrentDecoder", num_layers=2,),
)

# optimizer settings
# optimizer = dict(type="SGD", lr=1e-3, momentum=0.9, weight_decay=1e-2)
optimizer = dict(type="Adam", lr=1e-2)