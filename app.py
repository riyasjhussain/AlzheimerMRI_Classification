#!/usr/bin/env python
# coding: utf-8




import streamlit as st
import tensorflow.keras as keras
from PIL import Image
from tensorflow.keras.utils import load_img,img_to_array
import numpy as np

import keras.utils as image
from tensorflow.keras.applications.resnet import preprocess_input
model = keras.models.load_model("mritest.h5")
st.set_page_config(
   
    page_icon=":art:",
    layout="wide",
    initial_sidebar_state="expanded"
)

target_size = (176, 208)
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    image = image.resize(target_size)
    image_array = np.array(image)
    
    image_array = np.expand_dims(image_array, axis=0)
    # input_arr=np.expand_dims(input_arr,axis=0)
    y_predict=np.argmax(model.predict(image_array))
    y_predict
    if y_predict==0:
        st.write("MildDemented")
        
    elif y_predict==1:
         st.write("ModerateDemented")
    elif y_predict==2:
         st.write("NonDemented")
    else:
         st.write("VeryMildDemented")



