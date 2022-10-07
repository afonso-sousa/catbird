# general config settings
output_path = "./work_dirs"
num_workers = 4

# dataset settings
dataset_name = "Quora"
data_root = "data/quora/"
data = dict(
    max_length=40,
    train=dict(dataset_length=-1),
    val=dict(dataset_length=-1),
)

tokenizer = dict(name="roberta-base")

# train settings
train = dict(
    num_epochs=100,
    batch_size=32,
    accumulation_steps=1,
    with_amp=False,
    epoch_length=None,
    validation_interval=10,  # epochs
)

test = dict(
    print_output_every=15,  # batchs
    num_beams=1,
)

# model settings
model = dict(
    type="StackedResidualLSTM",
    encoder=dict(
        type="RecurrentEncoder",
        mode="LSTM",
        num_layers=3,
        hidden_size=512,
        dropout_out=0.5,
        residual=True,
    ),
    decoder=dict(
        type="RecurrentDecoder",
        mode="LSTM",
        num_layers=3,
        hidden_size=512,
        dropout_out=0.5,
        residual=True,
    ),
)

# optimizer settings
optimizer = dict(type="SGD", lr=8e-4, momentum=0.9)
# optimizer = dict(type="Adam", lr=8e-4)
# optimizer = dict(type="RMSprop", lr=8e-4)

# scheduler settings
scheduler = dict(num_warmup_epochs=4, peak_lr=1e-3)
