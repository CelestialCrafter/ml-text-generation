import tensorflow as tf
from data import get_dataset

dataset, vocab_size, (idx2char, char2idx) = get_dataset()

def generate_text(model, input_text, amount):
	states = None
	next_char = tf.constant([input_text])
	result = [next_char]

	for i in range(amount):
		next_char, states = model.generate_one_step(next_char, states=states)
		result.append(next_char)

	return tf.strings.join(result)[0].numpy().decode('utf-8')

if __name__ == '__main__':
	model_name = input('Model Name: ')
	print(generate_text(tf.saved_model.load(f'models/{model_name}'), input('Input Text: '), 1000))