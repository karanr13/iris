import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Streamlit app title
st.title("Iris Flower Classification Model")

st.image("E:\python\GIF.gif", caption='A beautiful Iris flower', use_container_width=True)

# Sidebar for user input
st.sidebar.header('Input Parameters')
st.sidebar.write("Adjust the sliders to set the values for the parameters of the iris flower.")
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (Cm)', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width (Cm)', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length (Cm)', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width (Cm)', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

# Gets user input and display it
df = user_input_features()
st.subheader('Inputed parameters')
st.write(df)

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Creates and trains the RandomForestClassifier model
clf = RandomForestClassifier()
clf.fit(X, Y)

# Makes predictions using the user input data
prediction = clf.predict(df.values)  # Converts DataFrame to NumPy array using .values
prediction_proba = clf.predict_proba(df.values)  # Converts DataFrame to NumPy array using .values

# Displays the class labels and their respective index numbers
st.subheader('Labels of Class and their respective index numbers')
st.write(iris.target_names)

# Displays the predicted iris flower species and confidence level
st.subheader('Predicted Iris Flower')
predicted_species = iris.target_names[prediction][0]
confidence = max(prediction_proba[0]) * 100
st.write(iris.target_names[prediction])

# Visualization of Predicted Probabilities
st.subheader('Predicted Probabilities for Each Iris Flower')
proba_df = pd.DataFrame(prediction_proba, columns=iris.target_names)
st.bar_chart(proba_df)

# Footer with link
link = 'Created by [Gideon Ogunbanjo](https://gideonogunbanjo.netlify.app)'
st.markdown(link, unsafe_allow_html=True)