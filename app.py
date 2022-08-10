import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from keras.preprocessing import image


from keras_preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
import json
import re

pickle_in = open("concrete","rb")
classifier=pickle.load(pickle_in)
st.write("""
         # Enhance your Image :) 
         """
         )
st.write("give your image a new look , choose your filters and see the result")
st.image("black-and-white-blue-filter.jpg", use_column_width=True)

st.sidebar.image("feedback.jpg", use_column_width=True)
add_selectbox = st.sidebar.selectbox(
    "Please elect your  filter",
    ("Select","Black and white ", "Blue ", "Red")
)

if add_selectbox == "Select":
    pass
else:
    st.sidebar.image("thanks_feedback.png", use_column_width=True)


file = st.file_uploader("Please upload an image file and select the filter from the", type=["jpg", "png"])


from PIL import Image, ImageOps


def import_and_predict(images, model):
        size = (64,64)    
        img = ImageOps.fit(images, size, Image.ANTIALIAS)
        img = np.asarray(img)
        img = np.expand_dims(img, axis=0)
        img = np.vstack([img])
        
        prediction = model.predict(img)        
        return prediction
    
      
    
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    classes = import_and_predict(image, model)

    if classes[0][0]==1:
        Output='Non Cracked'
    else:
        Output="Cracked"
        st.image("crack.jpg", width=300)
    
    st.write("Prediction : ",Output)

x = st.text_input('Please share a review')


#Sentiment Input
with open('tokenizer.json') as f:
	data = json.load(f)
	tokenizer = tokenizer_from_json(data)


TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
	return TAG_RE.sub('', text)

def preprocess_text(sen):
	# Removing html tags
	sentence = remove_tags(sen)

	# Remove punctuations and numbers
	sentence = re.sub('[^a-zA-Z]', ' ', sentence)

	# Single character removal
	sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

	# Removing multiple spaces
	sentence = re.sub(r'\s+', ' ', sentence)

	return sentence

vocab_size = len(tokenizer.word_index) + 1
maxlen = 256

X =[]

X.append(preprocess_text(x))
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X, padding='post', maxlen=maxlen)

if x=="":
    pass
else:

    y = model_sentimant.predict(X)


    Y = np.round(y)[0]


    if Y == 0:
	    st.write("Negative Review")
    else :
	    st.write("Positive Review")



