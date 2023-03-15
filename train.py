import tensorflow as tf

from keras.callbacks import EarlyStopping, BackupAndRestore

from models import OneStep, RNNModel
from generate import generate_text
from data import get_dataset
from callbacks import TextGenerationCallback

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)

dataset, vocab_size, ids_from_chars, chars_from_ids = get_dataset()

BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = (
	dataset
		.shuffle(BUFFER_SIZE)
		.batch(BATCH_SIZE, drop_remainder=True)
		.prefetch(tf.data.experimental.AUTOTUNE)
)

model_name = input('Model Name: ')
input_text = input('Input Text: ')

epochs = 100

model = RNNModel(vocab_size, 1024, 1024)
onestep = OneStep(model, chars_from_ids, ids_from_chars)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(
	dataset,
	epochs=epochs,
	callbacks=[
		EarlyStopping(monitor='accuracy', patience=1),
		BackupAndRestore(backup_dir=f'./models/checkpoint_{model_name}'),
		TextGenerationCallback(onestep)
	]
)


print(f'\n{generate_text(onestep, input_text, 100)}\n')

tf.saved_model.save(onestep, f'models/{model_name}')
