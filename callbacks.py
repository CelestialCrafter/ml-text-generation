from keras.callbacks import Callback
from generate import generate_text

class TextGenerationCallback(Callback):
	def __init__(self, customModel=None):
		self.customModel = customModel
	def on_epoch_begin(self, epoch, logs=None):
		print('\n\n' + generate_text(self.customModel or self.model, 'hello', 40))