import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

#load the model 
disease_model = tf.keras.models.load_model(r'D:\Python\python_project\tea_api\model_weights\tea_blister_blight_and_healthly_model')
disease_model_stem_cancer = tf.keras.models.load_model(r'D:\Python\python_project\tea_api\model_weights\tea_stem_and_bark_and_healthly_model')


def process_image_disease(image_path):
    image=cv2.imread(image_path)
    img_height,img_width= 224, 224
    image_resized= cv2.resize(image, (img_height,img_width))
    image=np.expand_dims(image_resized,axis=0)
    
    pred=disease_model.predict(image)
    class_names = [ "blister_blight" , "healthy"]
    output_class=class_names[np.argmax(pred)]
    probability =float(str(round(max(pred[0]),4)))
    return {"class":output_class ,"probability":probability}

def process_image_stem_cancer(image_path):
    image=cv2.imread(image_path)
    img_height,img_width= 224, 224
    image_resized= cv2.resize(image, (img_height,img_width))
    image=np.expand_dims(image_resized,axis=0)
    
    pred=disease_model_stem_cancer.predict(image)
    class_names = ["bark_cancer" , "healthy" , "leaf_cancer"]
    output_class=class_names[np.argmax(pred)]
    probability =float(str(round(max(pred[0]),4)))
    return {"class":output_class ,"probability":probability}