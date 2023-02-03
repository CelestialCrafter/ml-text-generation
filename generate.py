import tensorflow as tf

def generate_text(model, input_text, amount):
	for i in range(amount):
		input_text += ' ' + model.generate_word(input_text)
	return input_text

if __name__ == '__main__':
	model_name = input('Model Name: ')
	print(generate_text(tf.saved_model.load(f'models/{model_name}'), input('Input Text: '), 100))