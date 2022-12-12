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
    num_epochs=50,
    batch_size=32,
    accumulation_steps=1,
    with_amp=False,
    epoch_length=200,
    validation_interval=1,  # epochs
)

test = dict(
    print_output_every=15,  # batchs
    num_beams=1,
)

# model settings
model = dict(
    type="VanillaTransformer",
    encoder=dict(
        type="TransformerEncoder",
        embedding_size=512,
        num_heads=8,
        num_layers=3,
        ffnn_size=512,
        dropout=0.1,
    ),
    decoder=dict(
        type="TransformerDecoder",
        embedding_size=512,
        num_heads=8,
        num_layers=3,
        ffnn_size=512,
        dropout=0.1,
    ),
)

# optimizer settings
optimizer = dict(type="Adam", lr=1e-1)

# scheduler settings
scheduler = dict(num_warmup_epochs=2, peak_lr=1e-2)
