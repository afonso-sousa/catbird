# general config settings
output_path="./work_dirs"
print_output_every=50
num_workers=4

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
    num_epochs=70,
    batch_size=32,
    learning_rate=1e-4,
    weight_decay=1e-2,
    epoch_length=500,
    accumulation_steps=1,
    with_amp=False,
)

# model settings
model=dict(
  type="EDD",
  encoder=dict(
      type='RecurrentEncoder',
      mode="LSTM",
      embedding_size=512,
      hidden_size=512,
      dropout=0.5,
      num_layers=1,
  ),
)