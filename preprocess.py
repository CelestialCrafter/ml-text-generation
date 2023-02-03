import json
import re

with open(
	'data/dataset.json',
	encoding='ascii',
	errors='ignore'
) as f:
	dataset = json.load(f);

dataset = [f'{message["author"]["name"]}: {message["content"]}' for message in dataset['messages']]
dataset = [re.sub(r'((https?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&\'\(\)\*\+,;=.]+)', '', message) for message in dataset]
dataset = [re.sub(r'[^\w\s\-=_.?<>\'";:|]', '', message) for message in dataset]
dataset = list(filter(lambda message: message != '', dataset));

with open(
	'data/dataset.txt',
	'w',
	encoding='utf-8',
	newline='\n',
	errors='ignore'
) as f:
	f.write('\n'.join(dataset));
