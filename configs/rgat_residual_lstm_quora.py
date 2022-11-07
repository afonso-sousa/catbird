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
    with_dep=True,
)

tokenizer = dict(name="roberta-base")

# train settings
train = dict(
    num_epochs=100,
    batch_size=32,
    accumulation_steps=1,
    with_amp=False,
    validation_interval=5,  # epochs
)

test = dict(
    print_output_every=15,  # batchs
    num_beams=1,
)

# model settings
model = dict(
    type="RGATLSTM",
    # graph_layer=dict(
    #     coref=True),
    graph_encoder=dict(type="RGATEncoder", in_channels=256, hidden_size=256),
    encoder=dict(
        type="RecurrentEncoder",
        mode="LSTM",
        num_layers=2,
        hidden_size=256,
        dropout_out=0.5,
        residual=True,
    ),
    decoder=dict(
        type="RecurrentDecoder",
        mode="LSTM",
        num_layers=2,
        hidden_size=256,
        encoder_output_units=256,
        dropout_out=0.5,
        residual=True,
        fusion=True,
    ),
)

# optimizer settings
optimizer = dict(type="Adam", lr=1e-3)

# scheduler settings
scheduler = dict(num_warmup_epochs=2, peak_lr=1e-2)
