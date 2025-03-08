import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Loading Dataset
mnist = tf.keras.datasets.mnist

# Train and Test Split
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizing the data
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# Using a Sequential Model
model = tf.keras.models.Sequential()
# Flattening the layers to make one layer from 28 layers (Input Layer)
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

# Hidden Layers using ReLu as activation function and 128 nodes
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))


# Output Layer using Softmax activation function and 10 node for output of 10 individual digitis
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

# Compiling the model by choosing the optimizer, loss and mertics. 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=5)

#model.save('handwritten.keras')

#Once model is saved locally, above code can be commented and the below code can be used to run the model
#model = tf.keras.models.load_model('handwritten.keras')

image_number = 0
while os.path.isfile(f"Digits/{image_number}.png"):
    try:
        img = cv2.imread(f"Digits/{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()
    except:
        print("Error")
    finally:
        image_number += 1
# Evalutae the test data
#model.evaluate(x_test,y_test)