# general config settings
output_path = "./work_dirs"
num_workers = 4

# dataset settings
dataset_name = "Quora"
data_root = "data/quora/"
data = dict(
    mask_pad_token=True,
    max_length=50,
    train=dict(dataset_length=-1),
    val=dict(train_test_split=0.3, dataset_length=2000,),
)

# train settings
train = dict(
    num_epochs=100,
    batch_size=128,
    accumulation_steps=1,
    with_amp=False,
    epoch_length=100,
)

test = dict(print_output_every=5, num_beams=1,)

# model settings
model = dict(
    type="VanillaTransformer",
    encoder=dict(
        type="TransformerEncoder",
        embedding_size=256,
        num_heads=8,
        num_layers=3,
        ffnn_size=512,
        dropout=0.1,
    ),
    decoder=dict(
        type="TransformerDecoder",
        embedding_size=256,
        num_heads=8,
        num_layers=3,
        ffnn_size=512,
        dropout=0.1,
    ),
)

# optimizer settings
optimizer = dict(type="SGD", lr=0.1, momentum=0.9)
# optimizer = dict(type="Adam", lr=1e-2)

# scheduler settings
scheduler = dict(num_warmup_epochs=4, peak_lr=0.4)
