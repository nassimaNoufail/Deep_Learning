import numpy as np
import cv2
import os
import h5py
import nothin
import pandas as pd
import glob
from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

folders = glob.glob('../data/*')

y = []
x = []

x_test = []
classes = ['dhaval','prajeet','purvil','rajat']
for fol in classes:
	files = glob.glob('/Users/svdj16/Documents/6_sem_mini_project/CI_Innovative/test_face/'+fol+'/*')
	for f in files:
		print f
		img = cv2.imread(f)
		#img = img.flatten()
		x_test.append(img)
x_test = np.array(x_test)
print len(x_test), "hiiiiiiiiiiiiiiii"


classes = ['dhaval','prajeet','purvil','rajat']
for fol in classes:
	files = glob.glob('/Users/svdj16/Documents/6_sem_mini_project/CI_Innovative/augment_new/'+fol+'/*')
	for f in files:
		print f
		img = cv2.imread(f)
		#img = img.flatten()
		x.append(img)
	y+=[classes.index(fol) for i in range(len(files))]
	
x = np.array(x)
y = np.array(y)

print len(y)
x,y = shuffle(x,y)
y=MultiLabelBinarizer().fit_transform(y.reshape(-1, 1))
print x.shape
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)
print "Model Loaded......"
features = model.predict(x)
features_test = model.predict(x_test)

model = Sequential()
model.add(Flatten(input_shape=features.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
sgd = SGD(lr=0.00005, decay = 1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [ EarlyStopping(monitor='val_loss', patience=10, verbose=0), ModelCheckpoint('augment_1_1024.h5', monitor='val_loss', save_best_only=True, verbose=0) ]
model.load_weights('augment_1_1024.h5')
pred = model.predict(features_test)
print pred
#model.fit(features, y,validation_split=0.2, batch_size=32, nb_epoch=50,callbacks = callbacks)																#98% accuracy
#model.save_weights('weights_try_pre_2048.h5')


'''
model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(224, 224, 3)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

model.fit(x, y, batch_size=5, nb_epoch=100)
model.save_weights('weights_try_cnn.h5')
'''


#model = Sequential()
#model.add(Dense(input_dim=x.shape[1], output_dim=50, init='uniform', activation='tanh'))
#model.add(Dense(input_dim=50, output_dim=50, init='uniform', activation='tanh'))
#model.add(Dense(input_dim=50, output_dim=50, init='uniform', activation='tanh'))
#model.add(Dense(input_dim=50, output_dim=y.shape[1], init='uniform', activation='softmax'))
#sgd = SGD(lr=0.001, decay=1e-6, momentum=.9)
#model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
#model.load_weights('weight.h5')

#model.load_weights('weights_try.h5', by_name=True)
#model.fit(x,y,nb_epoch=300, batch_size=2)
#model.save_weights('weights_try.h5')
#answer = model.evaluate(x,y,verbose=0)
#scores = model.evaluate(x_test,y_test, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

