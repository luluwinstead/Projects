"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import datetime
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
num_epochs =200
csv_file = 'winequality-white.csv'
num_cols = 11
num_cols_after = 11
layer1_nodes = [105, 110, 165,285,315]
layer2_nodes = [0,5,55]#,120]
layer3_nodes = [0,50,65]
layer4_nodes=[0,15]
act_function = 'tanh'
activation2='sigmoid'
optimizer='RMSprop'
set_verbose = 2
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# load dataset
dataframe = pd.read_csv(csv_file, delimiter =";", header = 0)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
dataframe = dataframe.replace(np.nan,0)
dataset = dataframe.values

X = dataset[:,0:num_cols]
Y = dataset[:,num_cols]

X_MinMax = preprocessing.MinMaxScaler()
Y_MinMax = preprocessing.MinMaxScaler()
Y=np.array(Y).reshape((len(Y),1))
X=X_MinMax.fit_transform(X)

scaler=preprocessing.MinMaxScaler()

encoder=LabelEncoder()
encoder.fit(Y)
encoded_Y=encoder.transform(Y)
Y=to_categorical(encoded_Y)
Y=Y_MinMax.fit_transform(Y)

print(X.shape)
print(Y.shape)

X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.2)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#start_time = datetime.datetime.now()

ff=open('results.csv', 'w+')
header=('Layers' + ',' + 'l1n nodes' + ',' + 'l2n nodes' + ',' + 'Accuracy'+',' + 'Act_Function'+','+'\n')
ff.write(header)
ff.close()

for l1n in layer1_nodes:
    for l2n in layer2_nodes:
        model = Sequential()
        model.add(Dense(l1n, input_dim = num_cols_after, activation = act_function))
        for l3n in layer3_nodes:
            model = Sequential()
            model.add(Dense(l2n, input_dim = num_cols_after, activation = act_function))
            for l4n in layer4_nodes:
                model = Sequential()
                model.add(Dense(l3n, input_dim = num_cols_after, activation = act_function))
                if l4n > 0:
                    model.add(Dense(l4n, activation = act_function))
                    model.add(Dense(7, activation = activation2))
        
        # compile module
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.summary()
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        Train Model Section
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # fix random number seed
        seed = 50
        np.random.seed(seed)
        
        estimator = model.fit(X, Y, epochs = num_epochs, validation_data = (X_test, Y_test), verbose = set_verbose)
        score= model.evaluate(X_test, Y_test, verbose=0)
        ff=open('results.csv', 'a+')
        results = ('Layers' + ',' + str(l1n) + ',' +str(l2n)+ ',' + str(score[1])+',' + act_function+ ','+'\n' )
        ff.write(results)
        ff.close()
        
        
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        Show output Section
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        pyplot.plot(estimator.history['accuracy'])
        pyplot.xlabel("Number of Epochs")
        pyplot.ylabel("Accuracy")
        pyplot.show
       

#stop_time = datetime.datetime.now()
#print ("Time required for training:",stop_time - start_time)
prediction=model.predict(X_test)
confuse=metrics.confusion_matrix(Y_test.argmax(axis=1),prediction.argmax(axis=1))
print(confuse)