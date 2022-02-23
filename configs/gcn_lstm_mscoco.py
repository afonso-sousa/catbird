# general config settings
output_path = "./work_dirs"
print_output_every = 50
num_workers = 4

# dataset settings
dataset_name = "MSCOCO"
data_root = "data/mscoco/"
data = dict(
    max_length=80,
    train=dict(dataset_length=-1),
    val=dict(train_test_split=0.3, dataset_length=2000,),
    use_ie_graph=True,
)

# train settings
train = dict(
    num_epochs=70,
    batch_size=32,
    learning_rate=1e-4,
    weight_decay=1e-2,
    epoch_length=500,
    accumulation_steps=1,
    with_amp=False,
)

# model settings
model = dict(
    type="GCNLSTM",
    # graph_layer=dict(
    #     coref=True),
    graph_encoder=dict(type="GCNEncoder"),
    encoder=dict(type="RecurrentEncoder", num_layers=2,),
    decoder=dict(type="RecurrentDecoder", encoder_hidden_dim=128,),
)

# optimizer settings
optimizer = dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001)
