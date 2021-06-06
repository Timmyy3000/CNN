import streamlit as st 
from PIL import Image



import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image 

st.title("Pet Classifier")

file_img = st.file_uploader("Upload an image of your pet Dog or Cat", type ="jpg")

if file_img is not None:
    image_ = Image.open(file_img)

    st.image(image_, caption='Uploaded Image.', use_column_width=True)

    test_image = image_.resize((64, 64))

    

    # Convert image to numpy array
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    #  Load model

    loaded_model = tf.keras.models.load_model('cnn_model')

    # Test model
    result = loaded_model.predict(test_image/255)

    possible_dog_responses = ['Aww, what a cute little doggy',
                                "That's definitely a sleek Dog",
                                "Cool Dog,hope he doesn't bite",
                                "Woof, woof! I speak fluent Dog-ish",
                                 "Those ears, that fur, those eyes. Yup, That's a good ol' dog",
                                 "Canine Confirmed, Dog in sight"]
    possible_cat_responses = ["An elegant Cat indeed",
                                "Meow, meow",
                                "That's one sleek looking Cat",
                                "8 Lives, Cats are awesome",
                                "I hope your Cat doesn't bite", 
                                "Want some lasagna, Kitty ?"]

    import random

    result = result[0][0]
    isDog = False
    isCat = True
    

    if result > 0.5 :
        n = random.randint(0,5)
        pred = "Dog"
        prediction_test = possible_dog_responses[n]
        isDog = True
        isCat = False


    else :
        n = random.randint(0,5)
        pred = "Cat"
        prediction_test = possible_cat_responses[n]
        isCat = True
        isDog = False

    st.subheader(pred)
    st.write(prediction_test)
    
    if isDog :
        dog_prob = ((result - 0.5 ) / 0.5 )* 100
        cat_prob = 100 - dog_prob

    if isCat :
        cat_prob = (( 0.5 - result ) / 0.5 )* 100
        dog_prob = 100 - cat_prob

    st.subheader("Probabilities")
    st.write(f"Dog : {dog_prob:.2f} %")
    st.write(f"Cat : {cat_prob:.2f} %")

