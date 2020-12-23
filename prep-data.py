import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import pickle
from tqdm import tqdm


data = pd.read_csv('data/listings.csv')

data['price'] = data['price'].replace('[\$,]', '', regex=True).astype(float)

Airbnb_data = data[data.price > 10]
Airbnb_data = Airbnb_data[Airbnb_data.reviews_per_month > .12]

Airbnb_data = Airbnb_data.sample(frac=1)


# gets a list of the images as well as the prices
def get_img():
  img_list = []
  price_list = []
  data_dict=()
  for i in tqdm(range(2000)):
    try:

      response = requests.get(Airbnb_data['picture_url'][i])
      img = Image.open(BytesIO(response.content)).resize([224,224])

      img = np.array(img) / 255.0 # makes imputs [0,1]
      if img.shape == (224, 224, 3):
        img_list.append(img)
        price_list.append(Airbnb_data.price[i])
    except (KeyError or OSError):
      pass
  return img_list, price_list


X, y = get_img()


with open("data/images.txt", "wb") as fp:   #Pickling
   pickle.dump(X, fp)

with open("data/labels.txt", "wb") as fp:  # Pickling
   pickle.dump(y, fp)

