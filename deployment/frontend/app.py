import json
import streamlit as st
import requests
import cv2
from PIL import Image, ImageOps
import numpy as np


# load column transformer


st.title("Brain Tumor Detection App")

file = st.file_uploader("Please upload a brain MRI file", type=["png","jpg","jpeg"])
st.set_option('deprecation.showfileUploaderEncoding', False)

if file is not None:
    img = Image.open(file)

    # preprocess the image
    size = (224,224)    
    img_data = ImageOps.fit(img, size, Image.ANTIALIAS)
    img_data = np.asarray(img_data)
    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    #img_reshape = img[np.newaxis, -1].tolist()
    img_data = np.expand_dims(img_data, axis=0).tolist()

    # inference
    URL = "http://backend-tumor-classifier-ml2.herokuapp.com/v1/models/brain_tumor_classifier:predict"
    param = json.dumps({
            "signature_name":"serving_default",
            "instances":img_data
        })
    r = requests.post(URL, data=param)

    if r.status_code == 200:
        res = r.json()
        if np.argmax(res['predictions'][0]) == 0:
            st.title("Glioma Tumor")
        elif np.argmax(res['predictions'][0]) == 1:
            st.title("Meningioma Tumor")
        elif np.argmax(res['predictions'][0]) == 2:
            st.title("No Tumor")
        elif np.argmax(res['predictions'][0]) == 3:
            st.title("Pituitary Tumor")
    else:
        st.title("Unexpected Error")
    
    st.image(img,use_column_width= True)
    # To See details
    file_details = {"filename":file.name, "filetype":file.type,
                    "filesize":file.size}
    st.write(file_details)

elif file is None:
    st.text("")
