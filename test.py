# backend/predict.py

import sys
import os
import numpy as np
import tensorflow as tf
import cv2
import traceback
import pickle
import warnings


def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path,compile= False)
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        traceback.print_exc()
        sys.exit(1)


# Generate image features using the CNN
def extract_image_features(model, image_path):
    image_features = {}
    
    #read image using cv2
    img = cv2.imread(image_path)
    #Change image color to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # preprocess Image to shape used by model
    img = cv2.resize(img, (224,224)).reshape(1,224,224,3)
    
    #model prediction
    pred = model.predict(img,verbose = 0).reshape(1,2048)


    #grab only name of image without path   
    img_name = image_path.split('/')[-1]

    
    #save pred array to dictionary
    image_features[img_name] = pred

    return image_features



# Generate a caption for the image using the LSTM model
def generate_caption(model, image_features):


    #import dicitonary 
    with open('./dictionary/inv_dict.pkl', 'rb') as f:
        inv_dict = pickle.load(f)

    with open('./dictionary/word_dict.pkl', 'rb') as p:
        word_dict = pickle.load(p)

    text_inp = ['startofseq']

    count = 0
    caption = ''
    while count < 25:
        count += 1

        encoded = []
        for i in text_inp:
            encoded.append(word_dict[i])

        encoded = [encoded]

        encoded = tf.keras.preprocessing.sequence.pad_sequences(encoded, padding='post', truncating='post', maxlen=36)


        prediction = np.argmax(model.predict([image_features, encoded],verbose = 0))

        sampled_word = inv_dict[prediction]

        if sampled_word == 'endofseq':
            break

        caption = caption + ' ' + sampled_word
            
       

        text_inp.append(sampled_word)

    return caption



    return caption

if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=FutureWarning)

    # image_path = sys.argv[1]
    image_path = "./dataset/images/" + sys.argv[1]
    #load resnet

    cnn = load_model("./models/inception_v3_model.h5")

    lstm = load_model("./models/model.keras")

    #extract image features

    img_feat=extract_image_features(cnn,image_path)

    #generate caption

    caption = generate_caption(lstm,img_feat)
    
    
    print(caption)
