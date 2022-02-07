import streamlit as st
import tensorflow as tf
import streamlit as st
import pickle
#model=pickle.load(open('/content/drive/MyDrive/kaggle dataset/effnet.sav','rb'))
import tensorflow as tf
import cv2
import numpy as np
import PIL
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
#import streamlit as st
def cnn(model,image_arr):
    size = (128, 128)
    image_arr = cv2.cvtColor(np.float32(image_arr), cv2.COLOR_RGB2GRAY)
    new_image = cv2.resize(image_arr, (128, 128))
    new_image = np.array(new_image).reshape(-1, 128, 128, 1)
    print(new_image.shape)
    new_image = new_image / 255.0
    prediction = model.predict(new_image)
    class_name = ["Glioma", "Meningioma", "No_tumor", "Pituitary"]
    class_name = ["Glioma", "Meningioma", "No_tumor", "Pituitary"]
    st.write(class_name[np.argmax(prediction)])

    return prediction

def vgg16(model,image_arr):
    image_arr=cv2.cvtColor(np.float32(image_arr), cv2.COLOR_RGB2BGR)
    new_image=cv2.resize(image_arr,(224,224))
    new_image= new_image.reshape(1,224,224,3)
    # prediction = model.predict(new_image)
    # class_name = ["Glioma", "Meningioma", "No_tumor", "Pituitary"]
    # st.write(class_name[np.argmax(prediction)])
    p = model.predict(new_image)
    p = np.argmax(p,axis=1)[0]

    if p==0:
        p='Glioma Tumor'
    elif p==1:
        p='no tumor'
    elif p==2:
        p='Meningioma Tumor'
    else:
        p='Pituitary Tumor'

    if p!=1:
        print(f'The Model predicts that it is a {p}')
    st.write(p)
    return p

    return p

def effnet(model,image_arr):
    image_arr = cv2.cvtColor(np.float32(image_arr), cv2.COLOR_RGB2BGR)
    new_image = cv2.resize(image_arr, (150, 150))
    new_image = new_image.reshape(1, 150, 150, 3)
    # prediction = model.predict(new_image)
    # print(prediction)
    # class_name = ["Glioma", "Meningioma", "No_tumor", "Pituitary"]
    # st.write(class_name[np.argmax(prediction)])
    p = model.predict(new_image)
    p = np.argmax(p,axis=1)[0]

    if p==0:
        p='Glioma Tumor'
    elif p==1:
        p='no tumor'
    elif p==2:
        p='Meningioma Tumor'
    else:
        p='Pituitary Tumor'

    if p!=1:
        print(f'The Model predicts that it is a {p}')
    st.write(p)
    return p



#@st.cache(allow_output_mutation=True)
def main():
    st.title("BRAIN TUMOR CLASSIFICATION")
    activities = ['Basic CNN', 'VGG16', 'EffinetB0','Home']
    option = st.sidebar.selectbox('Which model would you like to use?', activities)
    st.subheader(option)

  #  @st.cache(allow_output_mutation=True)
    cccnn=pickle.load(open('/content/drive/MyDrive/kaggle dataset/traditionalcnn.sav','rb'))
    vggg16=pickle.load(open('/content/drive/MyDrive/kaggle dataset/vvgg16.sav','rb'))
    efffnet=pickle.load(open('/content/drive/MyDrive/kaggle dataset/effnet.sav','rb'))
    # cccnn = tf.keras.models.load_model("Traditionalcnn/cnn.h5")
    # vggg16=tf.keras.models.load_model("vgg16/vgg16.h5")
    # efffnet=tf.keras.models.load_model("efficientnet/effnet.h5")
    file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
    if file is None:
       st.text("Please upload an image file")
    else:
      image_arr = Image.open(file)
      st.set_option('deprecation.showfileUploaderEncoding', False)
      if option=="Basic CNN":
          cnn(cccnn,image_arr)
      elif option=="VGG16":
          vgg16(vggg16,image_arr)
      elif option=="EffinetB0":
          effnet(efffnet,image_arr)






if __name__=='__main__':
    main()