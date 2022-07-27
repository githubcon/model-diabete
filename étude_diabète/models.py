import pandas as pd
import numpy as np
import pickle

import streamlit as st
from streamlit_option_menu import option_menu

#1.as sidebar menu
st.title('Welcome to Diabetes Prediction Application using Machine Learning Algorithms') 

nom_fichier = ["RandomForest", "LogisticRegression", "DecisionTrees"] 

### Creation d'un volet contenant les 3 modèles

with st.sidebar:
	selected = option_menu("Main Menu", nom_fichier)



if selected == nom_fichier[0]:
	st.title(f"You have selected {selected }")

if selected == nom_fichier[1]:
	st.title(f"You have selected {selected }")

if selected == nom_fichier[2]:
	st.title(f"You have selected {selected }")



#### Importation des differents fichiers
file1 = open('diabetes_prediction_forest.pkl', 'rb')
Forest = pickle.load(file1)
file1.close()

file2 = open('diabetes_prediction_logreg.pkl', 'rb')
LR = pickle.load(file2)
file2.close()

file3 = open('diabetes_prediction_tree.pkl', 'rb')
DT = pickle.load(file3)
file3.close()

## Importation de la base population

data = pd.read_csv("diabete_population.csv")
print(data)
  

 ### Recuperation des features :
age = st.number_input("Enter your age")
grossesses = st.number_input("Enter your grossesses")
insuline= st.number_input("Enter your insuline")

# ##### Normalisation des features################################
 
age_norm = (age - data['age'].mean())/data['age'].std() 
grossesses_norm = (grossesses - data['grossesses'].mean())/data['grossesses'].std()
insuline_norm = (insuline - data['insuline'].mean())/data['insuline'].std()


if(st.button('Predict Diabete')):
	if (selected == nom_fichier[0]):
		query = np.array([grossesses_norm, age_norm, insuline_norm])

		query = query.reshape(1, 3)
		print(query)
    	prediction = rf.predict(query)[0]
    	st.title("Predicted value " + str(prediction) + str(nom_fichier[0]) )

	elif (selected == nom_fichier[1]): 
		query = np.array([grossesses_norm, age_norm, insuline_norm])

		query = query.reshape(1, 3)
    	print(query)
    	prediction = rf.predict(query)[0]
    	st.title("Predicted value " + str(prediction) + str(nom_fichier[1])) 
		
	elif (selected == nom_fichier[2]): 
		query = np.array([grossesses_norm, age_norm, insuline_norm])

		query = query.reshape(1, 3)
    	print(query)
    	prediction = rf.predict(query)[0]
    	st.title("Predicted value " + str(prediction) + str(nom_fichier[2])) 


	

