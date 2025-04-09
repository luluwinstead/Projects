# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:14:33 2019
adapted from  https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/
https://www.pythoncentral.io/how-to-pickle-unpickle-tutorial/
@author: apblossom
"""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing.text import text_to_word_sequence

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
in_filename = 'sequences_v2_Short.txt'
num_input_words = 100

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# load doc into memory
def load_doc(in_filename):
	# open the file as read only
	file = open(in_filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

#raw_text = open(filename, encoding="utf8").read()
#raw_text = raw_text.lower()
doc = load_doc(in_filename)
#lines = doc.split('\n')
#lines = doc.splitlines
lines = text_to_word_sequence(doc)
#print(lines)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# separate into input and output
sequences = array(sequences)
X, y = sequences[:], sequences[:]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# define model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(num_input_words, return_sequences=True))
model.add(LSTM(num_input_words))
model.add(Dense(num_input_words, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Train Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# fit model
model.fit(X, y, batch_size=128, epochs=1)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# save the model to file
model.save('model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))














