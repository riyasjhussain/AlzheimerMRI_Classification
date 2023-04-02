#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from PIL import Image
from tensorflow.keras.utils import load_img,img_to_array
import numpy as np



import keras.utils as image
from tensorflow.keras.applications.vgg19 import preprocess_input
model = keras.models.load_model("mritest.h5")
st.set_page_config(
    page_title="MRImage Classifier",
    page_icon=":art:",
    layout="wide",
    initial_sidebar_state="expanded"
)

uploaded_file = st.file_uploader("Choose image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    
    
    img=image.load_img(uploaded_file,target_size=(176,208))

    i=image.img_to_array(img)
    i=preprocess_input(i)
    input_arr=np.array([i])
    input_arr = input_arr/255.0
    
    # input_arr=np.expand_dims(input_arr,axis=0)
    y_predict=np.argmax(model.predict(input_arr))
    y_predict
    if y_predict==0:
        st.write("MildDemented")
        
    elif y_predict==1:
         st.write("ModerateDemented")
    elif y_predict==2:
         st.write("NonDemented")
    
    

    else:
         st.write("verymild")






