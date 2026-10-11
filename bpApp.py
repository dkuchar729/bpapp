import streamlit as st
import pandas as pd
import pydeck as pdk
import joblib

st.set_page_config(layout="wide")

#------------------- data and model part

dfClusters = pd.read_csv("clusters.csv", sep=';')
      
model=joblib.load('rfAccidentCount.pkl') 

#------------------- model input part

st.title("Predikce počtu nehod")

with st.form("inputForm",border=False):    
    col1,col2,col3 = st.columns(3)

    with col1:
        maxSpeed = st.slider("Maximální rychlost", min_value=10, max_value=130, value=90, step=5)
        probeCount = st.number_input("Počet sond", min_value=0, max_value=20000, value=100, step=100)
        

    with col2:
        segmentLength = st.number_input("Délka úseku", min_value=1, max_value=20000, value=1000, step=100)
    submitButton = st.form_submit_button("Provést predikci",type="primary")
inputData=[[maxSpeed,segmentLength,probeCount]]

if submitButton:
    prediction = model.predict(inputData)[0]
    
    st.metric(
        label="Výsledek predikce", 
        value=f"{prediction:.2f}"
    )

st.divider()

#------------------- heatmap part

mapOn=st.toggle("Zobrazit heatmapu shluků", value=False)

heatmapLayer = pdk.Layer(
        "HeatmapLayer",
        data=dfClusters,
        get_position=["wgsX", "wgsY"], 
        radius_pixels=30, 
        intensity=0.5
    )

startView = pdk.ViewState(
        latitude=48.9745564,
        longitude=14.4743400,
        zoom=10)

deck=pdk.Deck(
            layers=[heatmapLayer],
            initial_view_state=startView,
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json")

if mapOn:  
    st.pydeck_chart(
        deck,
        height=400)

