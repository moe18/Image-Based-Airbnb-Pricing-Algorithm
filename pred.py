import numpy as np
from tensorflow import keras
from PIL import Image

cat_clf = keras.models.load_model("models/my_keras_model.h5")

u_img = Image.open('/Users/mordechaichabot/Downloads/test_web_app_image.jpg')
# We preprocess the image to fit in algorithm.
img = u_img.resize([224,224])
image = np.asarray(img) / 255
print(np.array([image]).shape)


print(cat_clf.predict(np.array([image])))