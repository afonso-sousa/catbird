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
    type="EDD",
    encoder=dict(
        type="RecurrentEncoder",
        mode="LSTM",
        embedding_size=512,
        hidden_size=512,
        dropout_in=0.5,
        num_layers=1,
    ),
    decoder=dict(
        type="RecurrentDecoder",
        mode="LSTM",
        hidden_size=512,
        dropout_out=0.5,
    ),
    discriminator=dict(
        type="RecurrentDiscriminator",
        mode="GRU",
        embedding_size=256,
        hidden_size=512,
        dropout=0.5,
        num_layers=1,
        out_size=512,
    ),
)

# optimizer settings
# optimizer = dict(type="RMSprop", lr=8e-4)
optimizer = dict(type="SGD", lr=8e-4, momentum=0.9)
