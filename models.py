import tensorflow as tf

from keras import Model
from keras.layers import Embedding, GRU, Dense
from tensorflow import SparseTensor

class RNNModel(Model):
	def __init__(self, vocab_size, embedding_dim, rnn_units):
		super().__init__(self)
		self.embedding = Embedding(vocab_size, embedding_dim)
		self.gru = GRU(rnn_units, return_sequences=True, return_state=True)
		self.dense = Dense(vocab_size)

	def call(self, inputs, states=None, return_state=False, training=False):
		x = inputs
		x = self.embedding(x, training=training)
		if states is None:
			states = self.gru.get_initial_state(x)
		x, states = self.gru(x, initial_state=states, training=training)
		x = self.dense(x, training=training)

		if return_state:
			return x, states
		else:
			return x

class OneStep(Model):
	def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
		super().__init__()
		self.temperature = temperature
		self.model = model
		self.chars_from_ids = chars_from_ids
		self.ids_from_chars = ids_from_chars

		skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
		sparse_mask = SparseTensor(
			values=[-float('inf')]*len(skip_ids),
			indices=skip_ids,
			dense_shape=[len(ids_from_chars.get_vocabulary())]
		)
		self.prediction_mask = tf.sparse.to_dense(sparse_mask)

	@tf.function
	def generate(self, inputs, states=None):
		input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
		input_ids = self.ids_from_chars(input_chars).to_tensor()
		predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)
		predicted_logits = predicted_logits[:, -1, :]
		predicted_logits = predicted_logits/self.temperature
		predicted_logits = predicted_logits + self.prediction_mask
		predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
		predicted_ids = tf.squeeze(predicted_ids, axis=-1)
		predicted_chars = self.chars_from_ids(predicted_ids)
		return predicted_chars, states