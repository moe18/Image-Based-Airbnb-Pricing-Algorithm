# importing the libraries
import streamlit as st
from PIL import Image
import numpy as np
import time
from tensorflow import keras

# loading the cat classifier model
cat_clf = keras.models.load_model("models/my_keras_model.h5")




# functions to predict image
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))

    return s


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute the probability of a cat being present in the picture

    Y_prediction = sigmoid((np.dot(w.T, X) + b))

    return Y_prediction


# Designing the interface
st.title("Airbnb Image Pricing App")
# For newline
st.write('\n')


st.title("Upload Image")

# Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)
# Choose your own image
uploaded_file = st.file_uploader(" ", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    u_img = Image.open(uploaded_file)
    st.image(u_img, 'Uploaded Image', use_column_width=True)
    # We preprocess the image to fit in algorithm.
    img = u_img.resize([224,224])
    image = np.asarray(img) / 255

    my_image = image


if st.button("Predict Price"):

    if uploaded_file is None:

        st.sidebar.write("Please upload an Image to Price")

    else:

        with st.spinner('Predicting ...'):

            prediction = cat_clf.predict(np.array([image]))
            time.sleep(2)
            st.success('Done!')

        st.write("Algorithm Predicts:", prediction[0][0])




