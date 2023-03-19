import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

dataset = pd.read_csv(r"C:\Users\KBSPIDER\Handwriting Character Recognition\A_Z Handwritten Data.csv")
dataset.head(10)

dataset.astype('float32')
dataset.rename(columns={'0':'label'}, inplace=True)
X = dataset.drop('label',axis = 1)
y = dataset['label']

alphabets="abcdefghijklmnopqrstuvwxyz"
letter_name=[]
[letter_name.append(i) for i in alphabets]
name_tag = pd.DataFrame(letter_name)

plt.figure(figsize=(10,5))
sns.histplot(y,kde=False)
plt.title("Data Set")
plt.xlabel("Alphabets")
plt.ylabel("Images")

np.random.seed(2)
for i in range(2):
    plt.imshow(X.iloc[np.random.randint(0,372449)].values.reshape(28,28),cmap='Greys')
    plt.show()

dataset = np.loadtxt(r"C:\Users\KBSPIDER\Handwriting Character Recognition\A_Z Handwritten Data.csv", delimiter=',')
X = dataset[:,0:784]
Y = dataset[:,0]
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.3, random_state=2)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
X_train = X_train / 255
X_test = X_test / 255
print("Train data shape: ", X_train.shape)
print("Test data shape: ", X_test.shape)

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer = Adam(learning_rate = 0.01), loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())

Train1=model.fit(X_train, Y_train,batch_size=128, epochs=5)
Test1=model.fit(X_test, Y_test,batch_size=128, epochs=5)

Train2=model.fit(X_train, Y_train,batch_size=128, epochs=5)
Test2=model.fit(X_test, Y_test,batch_size=128, epochs=5)

Train3=model.fit(X_train, Y_train,batch_size=128, epochs=5)
Test3=model.fit(X_test, Y_test,batch_size=128, epochs=5)

history = model.fit(X_train, Y_train, epochs=1,  validation_data = (X_test,Y_test))
print("The training accuracy is :", history.history['accuracy'])
print("The training loss is :", history.history['loss'])

metrics = ['accuracy', 'loss']
plt.figure(figsize=(10, 5))
for i in range(len(metrics)):
    metric = metrics[i]
    plt.subplot(1, 2, i+1)
    plt.title(metric) 
    plt_train1 = Train1.history[metric] 
    plt_test1 = Test1.history[metric]
    plt_train2 = Train2.history[metric]
    plt_test2 = Test2.history[metric] 
    plt_train3 = Train3.history[metric]
    plt_test3 = Test3.history[metric] 
    plt.plot(plt_train1, label='train1') 
    plt.plot(plt_train2, label='train2') 
    plt.plot(plt_train3, label='train3') 
    plt.plot(plt_test1, label='test1') 
    plt.plot(plt_test2, label='test2') 
    plt.plot(plt_test3, label='test3')
    plt.xlabel("Epochs")
    plt.ylabel("Percentile")
    plt.legend() 
plt.show()

Val=X_test[[3258]]
prediction=model.predict(Val) 
prediction
plt.imshow(Val.reshape(28,28),cmap='Greys')

alphabets="abcdefghijklmnopqrstuvwxyz"
list1=[]
[list1.append(i) for i in range(26)]
list2=[]
[list2.append(i) for i in alphabets]
dic = dict(zip(list1, list2))
Prediction=dic[np.argmax(prediction)]

plt.imshow(Val.reshape(28,28),cmap='Greys')
print("The Predicted answer is",Prediction,".")
print("The validation accuracy is :", history.history['val_accuracy'])
print("The validation loss is :", history.history['val_loss'])
