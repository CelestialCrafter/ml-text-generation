import tensorflow as tf

def get_dataset():
	text = open('dataset.txt', 'rb').read().decode(encoding='utf-8')
	vocab = sorted(set(text))
	example_texts = ['abcdefg', 'xyz']

	chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
	ids_from_chars = tf.keras.layers.StringLookup(
			vocabulary=list(vocab), mask_token=None)
	ids = ids_from_chars(chars)

	chars_from_ids = tf.keras.layers.StringLookup(
			vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
	chars = chars_from_ids(ids)

	tf.strings.reduce_join(chars, axis=-1).numpy()

	all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
	ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

	seq_length = 100
	sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

	def split_input_target(sequence):
			input_text = sequence[:-1]
			target_text = sequence[1:]
			return input_text, target_text
	dataset = sequences.map(split_input_target)
	vocab_size = len(ids_from_chars.get_vocabulary())
	
	return (dataset, vocab_size, (chars_from_ids, ids_from_chars))
