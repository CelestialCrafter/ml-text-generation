import tensorflow as tf

from keras.layers import StringLookup
from tensorflow.strings import unicode_split, reduce_join

def get_dataset():
	text = open('data/dataset.txt').read()

	vocab = sorted(set(text))

	ids_from_chars = StringLookup(
			vocabulary=list(vocab), mask_token=None)

	chars_from_ids = StringLookup(
			vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

	all_ids = ids_from_chars(unicode_split(text, 'UTF-8'))
	ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

	seq_length = 100
	sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

	def split_input_target(sequence):
		input_text = sequence[:-1]
		target_text = sequence[1:]
		return input_text, target_text

	dataset = sequences.map(split_input_target)
	vocab_size = len(ids_from_chars.get_vocabulary())

	return dataset, vocab_size, ids_from_chars, chars_from_ids