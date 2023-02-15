from keras.models import Sequential
from keras.layers import *

def ANN(X_train_pca, Y_label, name):
    print('Running ANN')
    num_person = X_train_pca.shape[0]//19
    model = Sequential()
    model.add(Input(shape=(512)))
    model.add(Dense(units= 64, activation="relu", kernel_initializer='uniform'))
    model.add(Dense(units= 26, activation="relu", kernel_initializer='uniform'))
    model.add(Dense(units= num_person, activation="softmax", kernel_initializer='uniform'))
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    
    model.fit(X_train_pca, Y_label, epochs=20, batch_size=2, verbose=False)
    file = 'PJ1\\model\\' + name
    model.save(file)
    print("Done ANN !!")
    print('__________________________________________________________')
    
def ANN_fn(X_train_pca, Y_label, name):
    print('Running ANN FaceNet')
    num_person = X_train_pca.shape[0]//19
    model = Sequential()
    model.add(Input(shape=(512)))
    model.add(Dense(units= 256, activation="relu", kernel_initializer='uniform'))
    model.add(Dense(units= 128, activation="relu", kernel_initializer='uniform'))
    model.add(Dense(units= 64, activation="relu", kernel_initializer='uniform'))
    model.add(Dense(units= num_person, activation="softmax", kernel_initializer='uniform'))
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    
    model.fit(X_train_pca, Y_label, epochs=20, batch_size=2, verbose=False)
    file = 'PJ1\\model\\' + name
    model.save(file)
    print("Done ANN FaceNet !!")
    print('__________________________________________________________')