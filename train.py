import tensorflow as tf

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.callbacks import EarlyStopping, BackupAndRestore

from onestep import OneStep
from generate import generate_text
from data import get_dataset

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

dataset, tokenizer, max_sequence_length = get_dataset()
total_words = tokenizer.num_words

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

model = Sequential()
model.add(Embedding(total_words, 1))
model.add(LSTM(1))
model.add(Dense(total_words, activation='softmax'))

model.build()
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(
	dataset,
	epochs=epochs,
	callbacks=[
		EarlyStopping(monitor='accuracy', patience=1),
		BackupAndRestore(backup_dir=f'./models/checkpoint_{model_name}')
	]	
)

one_step_model = OneStep(model, tokenizer, max_sequence_length)

print(f'\n{generate_text(one_step_model, input_text, 100)}\n')

tf.saved_model.save(one_step_model, f'models/{model_name}')
