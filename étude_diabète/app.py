import streamlit as st
import pandas as pd
import numpy as np
import pickle
  
st.title('Welcome to Diabetes Prediction Application using Machine Learning Algorithms') 

file1 = open('diabetes_prediction.pkl', 'rb')
rf = pickle.load(file1)
file1.close()


data = pd.read_csv("diabete_population.csv")

print(data)
  
#age = st.number_input("Enter your age")
age= st.slider("Select the level", 21, 71) 
st.text('Selected: {}'.format(age))
 
age_norm = (age - data['age'].mean())/data['age'].std() 

grossesses = st.number_input("Enter your grossesses")
grossesses_norm = (grossesses - data['grossesses'].mean())/data['grossesses'].std()

insuline= st.number_input("Enter your insuline")
insuline_norm = (insuline - data['insuline'].mean())/data['insuline'].std()


if(st.button('Predict Diabete')): 
    query = np.array([grossesses_norm, age_norm, insuline_norm])

    query = query.reshape(1, 3)
    print(query)
    prediction = rf.predict(query)[0]
    st.title("Predicted value " +
             str(prediction)) 