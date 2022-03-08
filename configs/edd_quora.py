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
    num_epochs=50,
    batch_size=8,
    learning_rate=1e-4,
    weight_decay=1e-2,
    epoch_length=500,
    accumulation_steps=1,
    with_amp=False,
)

# model settings
model = dict(
    type="EDD",
    encoder=dict(
        type="RecurrentEncoder",
        mode="GRU",
        embedding_size=512,
        hidden_size=512,
        dropout=0.5,
        num_layers=1,
    ),
    decoder=dict(
        type="RecurrentDecoder", mode="LSTM", hidden_dim=512, dropout_out=0.5,
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
optimizer = dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001)
