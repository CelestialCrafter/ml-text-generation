import tensorflow as tf

def generate_text(model, input_text, amount):
	states = None
	next_char = tf.constant([input_text])
	result = [next_char]

	for i in range(amount):
		next_char, states = model.generate(next_char, states=states)
		result.append(next_char)

	result = tf.strings.join(result)
	return result[0].numpy().decode('utf-8')

if __name__ == '__main__':
	model_name = input('Model Name: ')
	input_text = input('Input Text: ')
	model = tf.saved_model.load(f'models/{model_name}')

	print(generate_text(model, input_text, 100))
