from numpy import true_divide
import streamlit as st
from streamlit.proto.Progress_pb2 import Progress
import back
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio
import subprocess
from PIL import Image

# subprocess.call(["python", "back.py"])


import streamlit as st

nav = st.sidebar.radio("Navigation", ["Predict", "Knowledge"])

if nav == "Predict":
    st.header("*CROP PREDICTION AND ANALYSIS*")
    st.image("cereal-grains-crops.jpg")
    
    i1 = st.selectbox("Location",["Bangalore","Chennai","Delhi","Lucknow","Mumbai","Jodhpur","Bhubhneshwar","Rourkela"],index=0)
    i2 = st.slider("Start Month", min_value = 1, max_value = 12, value = 1)
    i3 = st.slider("End Month", min_value = i2, max_value = 12, value = 12)
    st.write("Your Input:", i1, i2, i3)
    if st.button("Predict & Analyze"):
        
        subprocess.call(["python", "back.py"])
        final_predicted_crops,recommended_crop_lst = back.pred_recommend(i1,i2,i3)
        final_predicted_crops = final_predicted_crops[0]
        st.header("**Main Crop:**")
        st.subheader(final_predicted_crops)

        # Add an image
        img_file = r"C:\Users\Owner\Tri Nit\Crop Images\{}.jpg".format(final_predicted_crops)
        image = Image.open(img_file)
        st.image(image, caption= final_predicted_crops)
        

        st.subheader("**Other Potential Crops:**")


        
        
        
        st.write(recommended_crop_lst)


        st.subheader("Cost Analysis of Potential Crop")
        st.plotly_chart(back.make_price_bar(list(recommended_crop_lst)))

        st.subheader("Soil Composition for given Crop:")
        st.plotly_chart(back.soil_comp(final_predicted_crops))

        



if nav == "Knowledge":
    st.title("Thank You for Using Our Website")
    st.text("We aim to help farmers by providing the best crops that can be grown to maximize") 
    st.text("their profits. All the user needs to do is select the location where he wishes to")
    st.text("grow his crops, as well as the starting month and ending month for the season.")
    st.text("The website will then output the top 5 crops that is suitable for growth considering")
    st.text("the soil type and price of the crops.")

    st.text("Data used by us to generate our output is given below.")
    st.text("Bangalore - Laterite Soil")
    st.text("Chennai - Alluvial Soil")
    st.text("Delhi - Black Soil")
    st.text("Lucknow - Alluvial Soil")
    st.text("Mumbai - Black Soil")
    st.text("Jaipur - Arid Soil")
    st.text("Bhubhneshwar - Red Soil")
    st.text("Rourkela - Laterite Soil")

    st.text("A visual representation of various types of soils in India is given below.")
    st.image("Map.jpg")

    st.text("We have also considered other factors of the soil such as Ph value, Nitrogen,")
    st.text("Phosphorus, and Potassium ratios, and also external factors such as rainfall and")
    st.text("Temperature.")

    st.text("We are capable of providing analysis for 22 crops, of which list is given below.")
    st.text("Rice, Maize, Blackgram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon,")
    st.text("Apple, Orange, Papaya, Coconut, Cotton, Jute, Chickpea, Kidneybeans, Pigeonpeas,")
    st.text("Mothbeans, and Mungbeans.")
    st.text("That's it for the Documentation, now try our model on the Predict Page!")