import numpy as np
from keras import Model
from keras.utils import pad_sequences

class OneStep(Model):
	def __init__(self, model, tokenizer, max_sequence_length):
		super().__init__()
		self.model = model
		self.tokenizer = tokenizer
		self.max_sequence_length = max_sequence_length

	def generate_word(self, input_text):
		token_list = self.tokenizer.texts_to_sequences([input_text])[0]
		token_list = pad_sequences([token_list], maxlen=self.max_sequence_length-1, padding='pre')

		model_predictions = self.model(token_list)[0].numpy()

		prediction_indexes = [i for i in range(len(model_predictions))]
		prediction_indexes = np.array(sorted(prediction_indexes, key=lambda i: model_predictions[i], reverse=True))

		model_predictions = np.flip(np.sort(model_predictions))
		normalized_chances = model_predictions / model_predictions.sum()

		prediction = np.random.choice(prediction_indexes, p=normalized_chances)

		for word, i in self.tokenizer.word_index.items():
			if i == prediction:
 				return word