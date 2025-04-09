"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import pandas
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Load dataset
dataframe=pandas.read_csv("housing2.csv", delim_whitespace=True, header=None)
dataset=dataframe.values
# split into input (X) and output (Y) variables
X=dataset[:,0:13]
Y=dataset[:,13]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#print('dataset.type:',  type(dataset))
x0=dataset[:,0:3]
#print('x0= ', x0)

x1=dataset[:,3]
#print(x1)

x2=dataset[:,4:8]
#print(x2)

x3=dataset[:,8]
#print(x3)

x4=dataset[:,9:13]
#print(x4)

result = np.column_stack((x0, x2))
result = np.column_stack((result, x4))
#print(result)

#scale the data
scaler=MinMaxScaler()
result_scaled=scaler.fit_transform(result)
#print(result_scaled)

x1_cat=to_categorical(x1)
#print(x1_cat, x1_cat.shape)

x3_cat=to_categorical(x3)
#print(x3_cat, x3_cat.shape)
encoder=LabelEncoder()
encoder.fit(x3)
encoded_x3=encoder.transform(x3)
dummy_x3=to_categorical(encoded_x3)
#print(dummy_x3, dummy_x3.shape)

result=np.column_stack((result_scaled, x1_cat))
X=np.column_stack((result,dummy_x3))
#print(X, X.shape)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
start_time = datetime.datetime.now()

#create model
model=Sequential()
model.add(Dense(52, input_dim=22, activation='relu'))
model.add(Dense(26, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics =['accuracy'])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Train Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
history=model.fit(X,Y, epochs=100, batch_size=5, verbose=0)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
stop_time = datetime.datetime.now()



print ("Time required for training:",stop_time - start_time)


# Visualize model history - diagnostic plot
plt.figure()
plt.plot(history.history['accuracy'], label='Training accuracy')
#plt.plot(history.history['val_accuracy'], label='Validation accuracy')
#plt.title('Training / validation accuracies')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Training loss')
#plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('training / validation loss values')
plt.ylabel('Loss value')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()