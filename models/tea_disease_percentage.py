import cv2
import numpy as np
import os
import joblib 
import PIL


k_means_model = joblib.load(r"D:\Python\python_project\tea_api\model_weights\tea_diease_damage_percentage_model.joblib")

def image_to_array(image_path):
  image_array = np.array(PIL.Image.open(image_path))
  height = image_array.shape[0]
  width = image_array.shape[1]
  band = image_array.shape[2]
  numpy_array = image_array.reshape(-1 , band)
  return height , width , numpy_array



def prediction_model(image_path):

  new_height, new_width, new_x = image_to_array(image_path)
  new_labels = k_means_model.predict(new_x)
  cluster_counts = np.bincount(new_labels)  # Count the number of pixels in each cluster
  total_pixels = new_height * new_width
  percentage_pixels = cluster_counts / total_pixels * 100 
  
  damage_precentage = round(percentage_pixels[1] , 2)
  return damage_precentage