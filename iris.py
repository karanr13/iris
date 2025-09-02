import streamlit as st
import pickle
import numpy as np
import os

# Define the species names for clarity
species_names = ['Setosa', 'Versicolor', 'Virginica']

st.title('Iris Species Prediction App 🌺')
st.write("This app predicts the species of an Iris flower based on its measurements.")

# Add the image of the iris flower directly from its URL
st.image("E:\python\iris_image.png", caption='A beautiful Iris flower', use_container_width=True)

# Check if the model file exists before trying to load it
if not os.path.exists('iris_model.pkl'):
    st.error("Model file 'iris_model.pkl' not found. Please run 'train_model.py' first.")
    st.stop()

# Load the trained model
with open('iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create input widgets in a sidebar for a cleaner layout
st.sidebar.header('Input Features')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 0.2)
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width,
    }
    features = np.array(list(data.values())).reshape(1, -1)
    return features

# Get user input
input_features = user_input_features()

# Make prediction
prediction = model.predict(input_features)
predicted_species = species_names[prediction[0]]

# Display results
st.header('Prediction')
st.success(f"The predicted Iris species is: **{predicted_species}**")

# Display the input features used for prediction
st.subheader('Input Features')
st.write(f"Sepal Length: {input_features[0][0]:.2f} cm")
st.write(f"Sepal Width: {input_features[0][1]:.2f} cm")
st.write(f"Petal Length: {input_features[0][2]:.2f} cm")
st.write(f"Petal Width: {input_features[0][3]:.2f} cm")

st.markdown("---")
st.caption("Built with Streamlit and scikit-learn")