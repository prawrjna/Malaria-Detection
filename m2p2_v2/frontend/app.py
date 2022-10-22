import pandas as pd
import streamlit as st
import requests
import json
import numpy as np
import tensorflow as tf
from keras import utils
from keras.preprocessing import image
from keras_preprocessing.image import load_img, img_to_array
from PIL import Image
#import io

st.set_page_config(page_title='Malaria RBC', page_icon='favicon.png',layout='wide')

# widget input
st.title("Malaria Infested RBC Classifier")
st.subheader('By Prajna')
st.markdown('---')
with st.form('imginput'):
    uploaded_file = st.file_uploader("Choose a png file", type='png', help='Please only upload .png file')
    submitted = st.form_submit_button("Classify")
    if submitted:
        st.image(uploaded_file)
        img = Image.open(uploaded_file)
        img = img.convert('RGB')
        img = img.resize([130,130], Image.NEAREST)
        x = image.img_to_array(img) # untuk ubah image kedalam array
        x = x/255.
        x_1 = np.expand_dims(x, axis=0) #Memperluas bentuk array misal 1D jadi 2D, 0 berarti baris/horizontal [[1, 2]]

        # input ke model
        input_data_json = json.dumps({
        "signature_name": "serving_default",
        "instances": x_1.tolist()
        })

        # inference
        URL = "http://tfserving-back-prajna.herokuapp.com/v1/models/model_1:predict"
        r = requests.post(URL, data=input_data_json)

        if r.status_code == 200:
            res = r.json()
            if res['predictions'][0][0] >= 0.5:
                st.write('This blood cell is Uninfected')
            else:
                st.write('This blood cell is infected by malaria')
        else:
            st.write('Error')
    else: st.write('Please insert only .png file')
