import streamlit as st 
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image 



st.title("Pet Classifier")

st.write("""
A CNN works by extracting features from images. This eliminates the need for manual feature extraction. The features are not trained! They’re learned while the network trains on a set of images. This makes deep learning models extremely accurate for computer vision tasks. CNNs learn feature detection through tens or hundreds of hidden layers. Each layer increases the complexity of the learned features.

----

### A CNN
- starts with an input image
- applies many different filters to it to create a feature map
- applies a ReLU function to increase non-linearity
- applies a pooling layer to each feature map
- flattens the pooled images into one long vector.
- inputs the vector into a fully connected artificial neural network.
- processes the features through the network. The final fully connected 
- layer provides the “voting” of the classes that we’re after.
- trains through forward propagation and backpropagation for many, many epochs. This repeats until we have a well-defined neural network with trained weights and feature detectors.

----
 
#### This CNN can distinguish between imgaes of Dogs and Cats

Test it Below :)
""")


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

    if isDog :
        if dog_prob >= 80 :
            st.subheader("Confident : Pretty sure I got it right XD")
        else :
            st.subheader("Unsure : Not really confident in my prediction :/")
    
    if isCat :
        if cat_prob >=  80 :
            st.subheader("Confident : Pretty sure I got it right XD")
        else :
            st.subheader("Unsure : Not really confident in my prediction :/")

