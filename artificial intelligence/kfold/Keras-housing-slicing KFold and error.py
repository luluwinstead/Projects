"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import datetime
import pandas
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from keras.utils import to_categorical
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# load dataset
dataframe=pandas.read_csv('.csv', delim_whitespace=True, header=None)
dataset=dataframe.values

# split into input (X) and output (Y) variables
X=dataset[:,0:13]
Y=dataset[:,13]

print('dataset.type', type(dataset))
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

x0=dataset[:,0:3]
#print(x0)

x1=dataset[:,3]
#print(x1)

x2=dataset[:,4:8]
#print(x2)

x3=dataset[:,8]
#print(x3)

x4=dataset[:,9:13]
#print(x4)

result=np.column_stack((x0, x2))
result=np.column_stack((result, x4))
#print(result)
#scale the data
scaler=MinMaxScaler()
result_scaled=scaler.fit_transform(result)
#print(result_scaled)

x1_cat = to_categorical(x1)
#print(x1_cat, x1_cat.shape)

x3_cat = to_categorical(x3)
#print(x3_cat, x3_cat.shape)

encoder=LabelEncoder()
encoder.fit(x3)
encoded_x3=encoder.transform(x3)
#print(encoded_x3)
dummy_x3=to_categorical(encoded_x3)
#print(dummy_x3, dummy_x3.shape) 

result=np.column_stack((result_scaled,x1_cat))
X=np.column_stack((result,dummy_x3))  
#print(X, X.shape)                    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
start_time = datetime.datetime.now()

#create model
def create_model():
    model=Sequential()
    model.add(Dense(22, input_dim=X.shape[1], activation = 'relu'))
    model.add(Dense(1, activation = 'linear'))
    
    #compile network
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Train Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
n_split=3

for train_index, test_index in KFold(n_splits=n_split, shuffle=True, random_state=None).split(X):
    x_train, x_test=X[train_index], X[test_index]
    y_train, y_test=Y[train_index], Y[test_index]
    model=create_model()
    history=model.fit(x_train, y_train, batch_size=5, epochs=50, verbose=0, validation_data=(x_test, y_test))
    print('Model Evaluation:', model.evaluate(x_test, y_test))
    plt.figure()
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Test loss')
    plt.title('Loss values')
    plt.ylabel('Loss value')
    plt.xlabel('Epoch')
    plt.legend(loc="upper left")
    plt.show()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
stop_time = datetime.datetime.now()
print ("Time required for training:",stop_time - start_time)


