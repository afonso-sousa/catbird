"""This is an implementation of paper `Neural Paraphrase Generation
    with Stacked Residual LSTM Networks <https://aclanthology.org/C16-1275/>`.
"""

# general config settings
output_path = "./work_dirs"
num_workers = 4

# dataset settings
dataset_name = "MSCOCO"
data_root = "data/mscoco/"
data = dict(
    max_length=50,
    train=dict(dataset_length=-1),
    val=dict(
        train_test_split=0.3,
        dataset_length=2000,
    ),
)

tokenizer = dict(name="roberta-base")

# train settings
train = dict(
    num_epochs=150,
    batch_size=128,
    accumulation_steps=1,
    with_amp=False,
    epoch_length=None,
    validation_interval=10,  # epochs
)

test = dict(
    print_output_every=15,
    num_beams=1,
)

# model settings
model = dict(
    type="RecurrentModel",
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

# scheduler settings
scheduler = dict(num_warmup_epochs=4, peak_lr=1e-3)
