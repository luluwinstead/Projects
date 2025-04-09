"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import string
import re
import gensim
from gensim.utils import simple_preprocess
from pickle import dump
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
lengthLimit = 100
#lengthLimit = 100
#lengthLimit = 400
#low sequence length appears to have worse accuracy
seqLength = 500
epochs=15
#optimizer = 'rmsprop'
optimizer = 'adam'
#optimizer = 'Nadam'
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#load data into a pandas dataframe
train = pd.read_csv('train.csv')
# column names in the df
train = train[['tweet','label']]
#length of dataframe 
length = len(train)
#start time
start_time = datetime.datetime.now()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
##cleaning tweets
def clean_tweets(tweets):
    #creates pattern objects to match based on patterns
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    #remove unwanted char lines 74-77
    tweets = url_pattern.sub(r'', tweets)
    tweets = re.sub('\S*@\S*\s?', '', tweets)
    tweets = re.sub('\s+', ' ', tweets)
    tweets = re.sub("\'", "", tweets)   
    return tweets

#stored clean_tweets
listTweets = []
#tweets to list
val = train['tweet'].values.tolist()
#iterate through tweet list and append
for n in range(len(val)):
    listTweets.append(clean_tweets(val[n]))

##def function to remove punctuations 
def remove_punctuation(phrase):
    for sentence in phrase:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  

#apply function and append to new list
cleanedTweets = list(remove_punctuation(listTweets))
length2 = len(cleanedTweets)
def rtoken(wd):
    return TreebankWordDetokenizer().detokenize(wd)

##append to numpy array
tweet = []
for x in range(len(cleanedTweets)):
    tweet.append(rtoken(cleanedTweets[x]))
tweet = np.array(tweet)

##convert to categorical and float type
Label = np.array(train['label'])
labels = tf.keras.utils.to_categorical(Label, 2, dtype="float32")

##tokenizer : integer encoding sequence for words
tokenizer = Tokenizer(num_words=seqLength)
#tonizer is relevant to the tweet numpy array -- fit
tokenizer.fit_on_texts(tweet)
#compile tokens 
sequences = tokenizer.texts_to_sequences(tweet)
# these sequences are created based on the parameter set to the maxLength
processed = pad_sequences(sequences, maxlen=seqLength)

##perform train/test split
X_train, X_test, Y_train, Y_test = train_test_split(processed,labels, random_state=0)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#define the model : 
    # This model is sequential, layer stack with one input and one output tensor
model = Sequential()
#one embedding layer is added and the sequence length is added to this layer 
model.add(layers.Embedding(seqLength, 20))
#model.add(Dropout(0.2))
#LSTM Layer added with tanh acitivation function
model.add(LSTM(20, recurrent_dropout=0.0, recurrent_activation="sigmoid", activation="tanh",use_bias=True, 
               unroll=False, return_sequences=False))
#model.add(Dropout(0.3))
#second LSTM layer identical to first
#model.add(LSTM(10, recurrent_dropout=0.0, recurrent_activation="sigmoid", activation="tanh",use_bias=True, 
              # unroll=False, return_sequences=False))
##third LSTM layer 
#model.add(LSTM(80, recurrent_dropout=0.0, recurrent_activation="sigmoid", activation="tanh",use_bias=True, 
                ##unroll=False))
##dropout is added which reduces overfitting, similar to recurrent dropout within the layers.
model.add(Dropout(0.5))
## dense layer is added
model.add(Dense(2, activation='sigmoid'))
##print the summary od the model to assess layers and parameters of the model
print(model.summary())
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Train Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
## model compilation includes setting the model optimizer, loss to categorical, and metrics to accuracy
model.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics=['accuracy'])
##this next line saves the best model
checkpoint = ModelCheckpoint("bestModel1.hdf5", monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto', period=1,save_weights_only=False)
##fit the model based on the train, test sata, epochs are set to epoch variable and validation is set
## to the test set data, callbacks set to checkpoint to save best model.fit() data
history = model.fit(X_train, Y_train, epochs=epochs,validation_data=(X_test, Y_test),callbacks=[checkpoint])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
##stop time
stop_time = datetime.datetime.now()
##print run time of the model for comparison
print ("Time required for training:",stop_time - start_time)

# evaluate model based on the accuracy of the test set
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
#print accuracy
print('Accuracy: %f' % (accuracy*100))

#Plot 1 : Training Loss and Num Epochs
plt.figure()
#plot the training loss with label set to training loss
plt.plot(history.history['loss'], label='Training loss')
## Plot title 
plt.title('Training loss over Epochs: Is the RNN Model Learning?')
## y label added
plt.ylabel('Loss value')
## xlabel added
plt.xlabel('Number of Epochs')
## legend added to the upper left corner
plt.legend(loc="upper left")
## show the plot
plt.show()

##plot 2: Training and validation accuracy
plt.figure()
#plot for the training accuracy
plt.plot(history.history['accuracy'], label='Training accuracy')
##plot for the validation accuracy
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
##validation accuracy title
plt.title('Assessing Validation Accuracy of RNN Model')
## y axis label
plt.ylabel('Accuracy')
# x axis label
plt.xlabel('Number of Epochs')
## legend is in the upper lft corner ; idenifies lines
plt.legend(loc="upper left")
#show the plot
plt.show()


## prediction for confusion matrix
predictions = model.predict(X_test)
## metrics for the confusion matrix and will be will through the heat map below
confuse = metrics.confusion_matrix(Y_test.argmax(axis=1), predictions.argmax(axis=1))
#show the confusion matrix
print(confuse)

##test pandas dataframe for phrase 1 used in testing 
test = pd.read_csv('test.csv')
#length of train dataframe
length = len(train)
test = test[['id','tweet']]

## pull a few arbitrary sample tweets from the test set to check accuracy

##Calling for the best model that has been produced 
bestModel = keras.models.load_model("bestModel1.hdf5")
#classify hate speech or not hate speech
sAnalysis = ['Hate Speech detection: NO.','Hate Speech Detection: YES']
phrase1 = tokenizer.texts_to_sequences([str(test['tweet'][4])])
test1 = pad_sequences(phrase1, maxlen=lengthLimit)
#print test 
print(sAnalysis[np.around(bestModel.predict(test1), decimals=0).argmax(axis=1)[0]])
print(test['tweet'][4])  

##Calling for the best model that has been produced 
bestModel = keras.models.load_model("bestModel1.hdf5")
#classify hate speech or not hate speech
sAnalysis = ['Hate Speech detection: NO.','Hate Speech Detection: YES']
phrase2 = tokenizer.texts_to_sequences([str(test['tweet'][25])])
test2 = pad_sequences(phrase1, maxlen=lengthLimit)
#print test 
print(sAnalysis[np.around(bestModel.predict(test1), decimals=0).argmax(axis=1)[0]])
print(test['tweet'][25]) 

##Calling for the best model that has been produced 
bestModel = keras.models.load_model("bestModel1.hdf5")
#classify hate speech or not hate speech
sAnalysis = ['Hate Speech detection: NO.','Hate Speech Detection: YES']
phrase3 = tokenizer.texts_to_sequences([str(test['tweet'][12])])
test3 = pad_sequences(phrase1, maxlen=lengthLimit)
#print test 
print(sAnalysis[np.around(bestModel.predict(test1), decimals=0).argmax(axis=1)[0]])
print(test['tweet'][12]) 

##Calling for the best model that has been produced 
bestModel = keras.models.load_model("bestModel1.hdf5")
#classify hate speech or not hate speech
sAnalysis = ['Hate Speech detection: NO.','Hate Speech Detection: YES']
phrase4 = tokenizer.texts_to_sequences([str(test['tweet'][42])])
test4 = pad_sequences(phrase1, maxlen=lengthLimit)
#print test 
print(sAnalysis[np.around(bestModel.predict(test1), decimals=0).argmax(axis=1)[0]])
print(test['tweet'][42]) 

##Calling for the best model that has been produced 
bestModel = keras.models.load_model("bestModel1.hdf5")
#classify hate speech or not hate speech
sAnalysis = ['Hate Speech detection: NO.','Hate Speech Detection: YES']
phrase5 = tokenizer.texts_to_sequences([str(test['tweet'][67])])
test5 = pad_sequences(phrase1, maxlen=lengthLimit)
#print test 
print(sAnalysis[np.around(bestModel.predict(test1), decimals=0).argmax(axis=1)[0]])
print(test['tweet'][67]) 

## create a heat map for a better visualization
y_pred = model.predict(X_test)
matrix = confusion_matrix(Y_test.argmax(axis=1), y_pred.argmax(axis=1), labels=[0,1])
df_cm = pd.DataFrame(matrix, range(2), range(2))
#label size
sns.set(font_scale=1.1) 
# font size
sns.heatmap(df_cm, annot=True, fmt='d',cmap='Blues',annot_kws={"size": 10}) 
plt.show()
