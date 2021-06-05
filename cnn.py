# # Convolutional Nueral Network


# import libraries

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.__version__

 
# ## Data Processing 
 
# ### Training Set


# Training set images augmentation to avoid overfitting 

train_datagen = ImageDataGenerator (
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2 ,
    horizontal_flip = True
)


training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

 
# ### Testing Set


test_datagen =  ImageDataGenerator ( rescale = 1./255)

test_set = test_datagen.flow_from_directory ('dataset/test_set',
                                            target_size = (64,64),
                                            batch_size = 32,
                                            class_mode= 'binary'
    )

 
# ## Building The CNN
 
# ### Initializing the CNN


from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

cnn = Sequential() 

 
# #### Layers


# 1st Convolutional Layer
cnn.add(Conv2D(filters=32, kernel_size = 3, activation ="relu", input_shape=(64,64,3)))



# Pooling 
cnn.add(MaxPool2D(pool_size = 2, strides = 2))



# 2nd Convolutional Layer
cnn.add(Conv2D(filters=32, kernel_size = 3, activation ="relu"))

# 2nd Pooling layers
cnn.add(MaxPool2D(pool_size = 2, strides = 2))



# Flattening Layer
cnn.add(Flatten())



# Full COnnection
cnn.add(Dense(units = 128, activation = "relu"))



# Final Output Layer
cnn.add(Dense(units = 1, activation = 'sigmoid'))

 
# ## Training The CNN
 
#  #### Compiling the CNN


cnn.compile(optimzer = "adma", loss="binary_crossentropy", metrics = ['accuracy'])



# training nad testing model

cnn.fit(x=training_set, validation_data = test_set, epochs = 25 )



# Single prediction

import numpy as np
from tensorflow.keras.preprocessing import image 
import matplotlib.pyplot as plt

test_image = image.load_img('dataset/single_prediction/dog3.jpg', target_size = (64,64))

plt.imshow(test_image)
plt.show()

# Convert image to numpy array
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

# Predict
result = cnn.predict(test_image/255)


if result[0][0] > 0.5 :
    prediction = 'dog'
else :
    prediction = 'cat'

print(prediction)

 
# ## Save Model


# save model

cnn.save('cnn_model')



#  Load model

loaded_model = tf.keras.models.load_model('cnn_model')

# Test model
result = loaded_model.predict(test_image/255)


if result[0][0] > 0.5 :
    prediction_test = 'dog'
else :
    prediction_test = 'cat'

print(prediction_test)






