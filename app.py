import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from PIL import Image


# Load data
gold = pd.read_csv('gld_price_data (1).csv')

# Split into x and y
x = gold.drop(['Date', 'GLD'], axis=1)
y = gold['GLD']
print(x.shape, "\n", y.shape)

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=2
)
print(x_train.shape, x_test.shape)

reg = RandomForestRegressor()
reg.fit(x_train, y_train)

pred = reg.predict(x_test)
score = r2_score(y_test, pred)


# web app
st.title('Gold Price Model')

img = Image.open('img.jpeg')
st.image(img, width=200)

st.subheader('Using randomforestregressor')
st.write(gold)

st.subheader('Model Performance')
st.write(score)
