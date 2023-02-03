# tensorflow-rnn

Simple RNN that I made for fun

Setup:
	1. Install [Tensorflow](https://www.tensorflow.org/install)
	2. Install numpy (`pip install numpy`)
	3. Create the `models` folder
	4. Create the `data` folder
	5. Create a text file named `data/dataset.txt` and put your dataset into it
	5a. If you used a tool like DiscordChatExporter, you can put your JSON into data/dataset.json and then run preprocess.py

Usage:
	1. Train the model with `train.py`; Input a model name and wait for the training to finish
	2. Generate text with `generate.py`; Input the model name and the seed text