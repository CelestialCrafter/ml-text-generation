import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_dataset():
	text = open('data/dataset.txt', 'rb').read().decode(encoding='utf-8')
	corpus = text.split('\n')
	corpus = [line + ' \n' for line in corpus]

	tokenizer = Tokenizer(num_words=2000)
	tokenizer.fit_on_texts(corpus)

	sequences = []
	for line in corpus:
		token_list = tokenizer.texts_to_sequences([line])[0]
		for i in range(1, len(token_list)):
			n_gram_sequence = token_list[:i+1]
			sequences.append(n_gram_sequence)

	max_sequence_length = max([len(seq) for seq in sequences])
	sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_length, padding='pre'))

	input_sequences, labels = sequences[:,:-1], sequences[:,-1]

	dataset = tf.data.Dataset.from_tensor_slices((input_sequences, labels))

	return dataset, tokenizer, max_sequence_length
