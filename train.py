import tensorflow as tf
from keras.callbacks import EarlyStopping, BackupAndRestore
from models import RNNModel, OneStep
from data import get_dataset
from generate import generate_text

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

dataset, vocab_size, (chars_from_ids, ids_from_chars) = get_dataset()

modelName = input('Model Name:')

BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = (
	dataset
		.shuffle(BUFFER_SIZE)
		.batch(BATCH_SIZE, drop_remainder=True)
		.prefetch(tf.data.experimental.AUTOTUNE)
)

embedding_dim = 256
rnn_units = 512
epochs = 20
model = RNNModel(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units)

for input_example_batch, target_example_batch in dataset.take(1):
	example_batch_predictions = model(input_example_batch)
	print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

model.fit(
	dataset,
	epochs=epochs,
	callbacks=[
		EarlyStopping(monitor='loss', patience=2),
		BackupAndRestore(backup_dir=f'./models/checkpoint_{modelName}')
	]
)

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
print(generate_text(one_step_model, 'I', 1000))

tf.saved_model.save(one_step_model, f'models/{modelName}')
