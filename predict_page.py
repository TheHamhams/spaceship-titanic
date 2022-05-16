from ctypes import create_unicode_buffer
import streamlit as st 
import pickle
import numpy as np
import pandas as pd

def load_model():
    with open('model.pickle', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data['model']
le_planet = data['le_planet']
le_destination = data['le_destination']
le_cabin = data['le_cabin']

def show_predict_page():
    st.title("Spaceship Titanic Prediction")
    
    st.write("""
             ### We need some information to predict if you were teleported.
              """)

    home_planets = (
        "Earth",
        "Europa",
        "Mars"
    )

    destinations = (
        "TRAPPIST-1e",
        "55 Cancri e",
        "PSO J318.5-22"
    )

    cabins = (
        'A',
        'B',
        'C',
        'D',
        'E',
        'F',
        'G',
        'T'
    )

    home_planet = st.selectbox('Your Home Planet', home_planets)
    destination = st.selectbox('Your Destination', destinations)
    cabin = st.selectbox('Your Cabin', cabins)
    
    age = st.slider("Age?", 0, 80, 27)
    
    cryosleep = st.checkbox('Did you use Cryo Sleep?')
    if cryosleep == True:
        cryosleep = 1
    else:
        cryosleep = 0
        
    food = st.checkbox('Did you use money at the food court?')
    if food == True:
        food = 1
    else:
        food = 0
    
    vip = st.checkbox('Were you a VIP?')
    if vip == True:
        vip = 1 
    else:
        vip = 0
    
    starboard = st.checkbox('Was your cabin on the starboard side?')
    if starboard == True:
        starboard = 1   
    else:
        starboard = 0
    
    ok = st.button("Predict Teleportation")
    
    if ok:
        X = np.array([[home_planet, cryosleep, destination, age, vip, food, cabin, starboard]])
        X[:, 0] = le_planet.transform(X[:,0])
        X[:, 2] = le_destination.transform(X[:,2])
        X[:, 6] = le_cabin.transform(X[:,6])
                
        
        if model.predict(X) == True:
            st.subheader('Uh oh, you were transported to another dimension!')
        elif model.predict(X) == False:
            st.subheader('Looks like you made it to your destination safely!')
        
        
