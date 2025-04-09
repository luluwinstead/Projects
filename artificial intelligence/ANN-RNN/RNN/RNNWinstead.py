"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import datetime
import matplotlib as plt
import tensorflow as tf
from tensorflow import keras
import string
import re
import gensim
import numpy
from keras.models import Model, load_model
from gensim.utils import simple_preprocess
from keras.optimizers import Adam
from pickle import dump
from matplotlib import pyplot
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tensorflow.keras.layers import Bidirectional
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, SimpleRNN, GRU
from sklearn import metrics

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
## how many words can be taken into the model at a given time
wordLimit =2000
## how much text can be run instantaneously 
lengthLimit =100
## number of epochs per run 
epochs=10
#batch size
batch_size=200
#validation split
validation_split=0.2
##wordLimit is one of the three parameters 
## Bigger one gets more accuracy
optimizer= 'adam'
#embedding layer
# embedding_layer=Embedding(1000,64)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
## create pandas dataframe and read the train csv file
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
## count the length of the train dataframe
length = len(train)
## create dataframe for tweet and associating label, 1 or 0 identifying hate 
##speech
train = train[['tweet','label']]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#a def function to clean the tweets for bettern compilation through the model 
def clean_tweets(tweets):
    #expression to remove url links.
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    ## parse tweets
    tweets = url_pattern.sub(r'', tweets)
    tweets = re.sub('\S*@\S*\s?', '', tweets)
    tweets = re.sub('\s+', ' ', tweets)
    tweets = re.sub("\'", "", tweets) 
    #convert to lowercase
    tweets = lowercase(tweets)
    #tweets=sorted(tweets)
    return tweets

def lowercase(tweets):
    ##created def function to lowercase
    return tweets.lower()

listTweets = []
##create list for train tweet values
convertList = train['tweet'].values.tolist()
for i in range(len(convertList)):
    listTweets.append(clean_tweets(convertList[i]))
    
def remove_punctuation(wordPhrase):
    for sentence in wordPhrase:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True)) 

## create list for tweets list that have had punctuation removed
cleanTweets = list(remove_punctuation(listTweets))
##count length of the list of clean tweets
length2 = len(cleanTweets) 

##definition function to detokenize
def rtoken(wd):
    return TreebankWordDetokenizer().detokenize(wd)

##create empty list for tweets to append clean tweets to
tweet = []
## for the range of the lenght of index of cleaned tweets, append the 
#x to tweet list
for i in range(len(cleanTweets)):
    tweet.append(rtoken(cleanTweets[i]))
    
# convert list to numpy array
tweet = np.array(tweet)
## create anumpy array for the label tweets
label = np.array(train['label'])
## convert the label tweets in the numpy array to categorical and set data type to float32 
##there are two categories
labels = tf.keras.utils.to_categorical(label, 2, dtype="float32")
tokenizer = Tokenizer(num_words=wordLimit) #convert text to a token
tokenizer.fit_on_texts(tweet)
#pad token tweet texts to sequences
sequences = tokenizer.texts_to_sequences(tweet)
vocab_size=len(tokenizer.word_index)+1
#set max length in the sequences
processed = pad_sequences(sequences, maxlen=vocab_size)
#perform train/ and test split, random state=0 

X_train, X_test, Y_train, Y_test = train_test_split(processed,labels, random_state=0)
#X_train = numpy.reshape(X_train, (X_train.shape[0], , X_train.shape[1]))
#X_test = numpy.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('x_train shape:', X_train.shape)
print('x_test shape:', X_test.shape)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('Build model...')
##begin the time  to assess model run time for comparison
start_time = datetime.datetime.now()


#experiment with embedding and
##LSTM(double and half) -> 
## add an additional dense layer

##Note: adding an additional layer makes it worse.
model = Sequential() # base set // stack -> only one input and output tensor
#embedding layer transforms the sequence of integers into a standard size
model.add(layers.Embedding(wordLimit, 20)) #embedding layer
##bidirectional layer
##model.add(layers.Bidirectional(layers.LSTM(32)))
#model.add(Bidirectional(LSTM(128, dropout=0.0, recurrent_dropout=0.0, recurrent_activation="sigmoid", activation="tanh",use_bias=True, unroll=False,return_sequences=True)))
#Can also use CuDNN
#model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.0, recurrent_activation="sigmoid", activation="tanh",use_bias=True, unroll=False,)))
##bidirectional exmple py file
##model.add(Embedding(max_features, 128))
#model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.0, recurrent_activation="sigmoid", activation="tanh",use_bias=True, unroll=False,)))
##stacking current layers
#model.add(SimpleRNN(32,return_sequences=True))
#model.add(SimpleRNN(32,return_sequences=True))
#model.add(SimpleRNN(32,return_sequences=True))
#model.add(SimpleRNN(32))
##GRU as alternate to LSTM solves vanishing gradient problem
#model.add(GRU(10, dropout=0.2, recurrent_dropout=0.0, recurrent_activation="sigmoid", activation="tanh",use_bias=True, unroll=False, reset_after=True, return_sequences=False))

#model.add(layers.LSTM(128,dropout=0.2, return_sequences=True))
#model.add(layers.LSTM(128,dropout=0.2))

model.add(layers.Dense(2,activation='softmax'))

#model.add(layers.LSTM(units=512, dropout=0.2, recurrent_dropout=0.0, recurrent_activation="sigmoid", activation="tanh",use_bias=True, unroll=False, return_sequences=True))
#model.add(layers.LSTM(units=512, dropout=0.2, recurrent_dropout=0.0, recurrent_activation="sigmoid", activation="tanh",use_bias=True, unroll=False))
#model.add(layers.Dense(units=10000, activation='relu'))
#model.add(layers.Dense(units=wordLimit, activation='softmax'))
#model.add(Dense(2,activation='softmax')) 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Train Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
## could also try binary with embedding matrix
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint("best_model1.hdf5", monitor='val_accuracy', verbose=1,save_best_only=True, 
                             mode='auto', period=1,save_weights_only=False)
print('Train...')
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, validation_data=(X_test, Y_test),callbacks=[checkpoint])
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

score, acc = model.evaluate(X_test, Y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
stop_time = datetime.datetime.now()
print ("Time required for training:",stop_time - start_time)

# save the model to file
#model.save('model3.h5')
# save the tokenizer
#dump(tokenizer, open('tokenizer_m3.pkl', 'wb'))
#stop_time = datetime.datetime.now()

plt.figure()
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of Epochs')
plt.legend(loc="upper left")
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.title('training / validation loss values')
plt.ylabel('Loss value')
plt.xlabel('Number of Epochs')
plt.legend(loc="upper left")
plt.show()

predictions = model.predict(X_test)
confuse = metrics.confusion_matrix(Y_test.argmax(axis=1), predictions.argmax(axis=1))
print(confuse)

