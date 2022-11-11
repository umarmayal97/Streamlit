import streamlit as  st
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

header=st.container()
dataset=st.container()
feature=st.container()
model_traing=st.container()

with header:
    st.title("Titanic App")
    st.markdown("This app is a simple web app to predict the survival of a passenger")

with dataset:
    st.header("Kashti doob gai")
    st.markdown("we are working on Titanic Dataset")
    df= sns.load_dataset("titanic")
    st.write(df.head(7))
    st.bar_chart(df["sex"].value_counts()) 
with feature:
    st.header("These are our App feature")

with model_traing:
    st.header("Model Training")
