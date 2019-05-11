#import pandas as pd
import numpy as np
import cv2
import os
import h5py
from keras.models import Model, load_model, Sequential
from keras.layers import Input, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard,EarlyStopping
from keras.utils.io_utils import HDF5Matrix
from keras.applications.densenet import *
from keras import backend as K
from keras.layers.normalization import BatchNormalization

SEQ_LEN = 10
def RNNmodel():
    input_shape = (SEQ_LEN, 1024)
    # Model
    
    print('Loading Model')
    model = Sequential()
    model.add(LSTM(2048,input_shape=input_shape,dropout=0.6))
#     model.add(BatchNormalization())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1024, activation='relu'))
#     model.add(BatchNormalization())
    model.add(Dropout(0.8))
    model.add(Dense(6, activation='softmax'))
    optimizer = Adam(lr=1e-5, decay=1e-6)
    metrics = ['accuracy', 'top_k_categorical_accuracy']
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=metrics)
    model.load_weights('_mobiletechniqueKTH-weight.hdf5')
    
    
    #model = load_model('Activity_Recognition.h5',compile=False)
    print('Done RNN')
    return model
    
def CNNmodel():
    print('Loading CNN')
    base_model = DenseNet121(weights='imagenet',include_top=True)
# We'll extract features at the final pool layer.
    model = Model(inputs=base_model.input,outputs=base_model.get_layer('avg_pool').output)
    
    return model

#y_prob = model.predict(x) 
#y_classes = y_prob.argmax(axis=-1)
#labels = ["boxing" , "handclapping" , "handwaving" , "jogging" , "running" ,"walking"]
#predicted_label = sorted(labels)[y_classes]



#RNNmodel.load_weights('models_checkpoint-01-2.23.hdf5')

K.clear_session()
CNNmodel = CNNmodel()
RNNmodel = RNNmodel()
def test():
    clip = []
    cap = cv2.VideoCapture('v_BaseballPitch_g01_c01.avi')
    for i in range(SEQ_LEN):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (224,224))
        clip.append(frame)

    clip = np.array(clip)
    features = CNNmodel.predict(clip)
    features = features[np.newaxis,::]
    prediction = RNNmodel.predict(features)
    print(prediction)

    labels = ["boxing" , "handclapping" , "handwaving" , "jogging" , "running" ,"walking"]
    prediction = prediction.tolist()
    i = 0
    for label in labels:
        print( label + "==>" + str(prediction[0][i]) )
        i = i + 1
    prediction = RNNmodel.predict_classes(features)
    print(prediction)

def start():
    cap = cv2.VideoCapture(0)
    print('here')
    ret = True

    clip = []
    while ret:
        #read frame by frame
        ret, frame = cap.read()


        # resize to the test size
    #    tmp_ = center_crop(cv2.resize(frame, (171, 128)))

        #seems normalize color
    #   tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        frame = cv2.resize(frame, (224,224))
        clip.append(frame)
        if len(clip) == SEQ_LEN:
            # 16 * 112 * 112 * 3
            inputs = np.array(clip).astype(np.float32)
            #print(inputs.shape)

            # 1 * 16 * 112 * 112 * 3
            #inputs = np.expand_dims(inputs, axis=0)
            #print(inputs.shape)
            # 1 * 3 * 16 * 112 * 112
            #inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            #print(inputs.shape)
            features = CNNmodel.predict(np.array(clip))
            features = features[np.newaxis,::]
            prediction = RNNmodel.predict(features)
            #print(prediction)
            print("\n\n\n\n")
            print("----------------------------------------------")
            labels = ["boxing" , "handclapping" , "handwaving" , "jogging" , "running" ,"walking"]
            prediction = prediction.tolist()
            i = 0
            for label in labels:
                print( label + "==>" + str(prediction[0][i]) )
                i = i + 1
            listv = prediction[0]
            n = listv.index(max(listv))
            print("\n")
            print("----------------------------------------------")
            print( "Highest Probability: " + labels[n] + "==>" + str(prediction[0][n]) )
            print("----------------------------------------------")
            print("\n")


            #prediction = RNNmodel.predict_classes(features)
            #print(prediction)
            
            #put into model
            clip = []

            cv2.imshow('result', frame)
            cv2.waitKey(30)

    cap.release()
    cv2.destroyAllWindows()




test()
start()