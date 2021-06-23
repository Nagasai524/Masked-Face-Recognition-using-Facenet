import streamlit as st
st.set_page_config(layout="centered")
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.spatial.distance import canberra
from sklearn.preprocessing import Normalizer

#loading the facenet model
model_emb=load_model('facenet.h5')

#Loading the mtcnn model
detector=MTCNN()

l2_encoder=Normalizer(norm='l2')

#Modify the argument of np.load() function.
#data=np.load('location to emebeddings numpy file that got generated using emebeddings_generator.py')
data=np.load('data.npz')
trainx_embed,trainy=data['a'],data['b']

def get_pixels(img):
  img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  results=detector.detect_faces(img)
  if not(results):
      return (0,np.array([1,2,3]))
  x1,y1,width,height=results[0]['box']
  x1,y1=abs(x1),abs(y1)
  x2,y2=x1+width,y1+height
  face=img[y1:y2,x1:x2]
  face=cv2.resize(face,(160,160))
  return ([x1,x2,y1,y2],face)

def get_embeddings(face_pixels):
    face_pixels=face_pixels.astype('float32')
    mean,std=face_pixels.mean(),face_pixels.std()
    face_pixels=(face_pixels-mean)/std
    samples=np.expand_dims(face_pixels,axis=0)
    yhat=model_emb.predict(samples)
    return yhat[0]

def calculate_distance(embedding,known_faces,known_labels):
  store=dict()
  for i in known_labels:
    if i not in store:
      store[i]=[]
  for i in range(known_faces.shape[0]):
    store[known_labels[i]].append(canberra(embedding,known_faces[i]))
  for i in store.keys():
    store[i]=sum(store[i])/len(store[i])
  dist=min(store.values())
  for i in store:
    if store[i]==dist:
      return (dist,i)

st.markdown("# <center> <u> Masked Face Recognition</u> </center> <br/> <br/>",True)
st.markdown("## <center> Upload an Image File </center>",True)
a=st.file_uploader("")
cnt1,cnt2=st.beta_columns(2)
if a:
    with cnt1:
      st.markdown("### <center>Uploaded Image</center>",True)
      st.image(a)
    image=Image.open(a)
    im=image.save('a.jpg')
    image=cv2.imread('a.jpg')
    cord,face_pixels=get_pixels(image)
    if cord==0:
        st.write("No Face detected in Image.")
    else:
        x1,x2,y1,y2=cord[0],cord[1],cord[2],cord[3]
        cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),10)
        face_embeddings=get_embeddings(face_pixels)
        face_embeddings_norm=l2_encoder.fit_transform([face_embeddings])
        trainx_norm=l2_encoder.fit_transform(trainx_embed)
        distance,label=calculate_distance(face_embeddings_norm,trainx_norm,trainy)
        print(label,distance)
        if distance>75:
            label="UNKNOWN"
        cv2.putText(image,label,(x1+80,y2+70),cv2.FONT_HERSHEY_COMPLEX,4,(0,255,255),8)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        with cnt2:
          st.markdown("### <center>Processed Image</center>",True)
          st.image(image)
          st.write(label,distance)
        st.success("The Recognized person in the image is "+label)
