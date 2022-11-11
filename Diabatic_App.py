import streamlit as  st
import pandas as pd
import sklearn
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

header=st.container()
dataset=st.container()
feature=st.container()
model_traing=st.container()

with header:
    st.header("Diamond Price App")
    
with dataset:
    st.header("Load DataSet")
    df= sns.load_dataset("diamonds")
    #st.write(df.info)
    
with feature:
    st.subheader("We have one Feature : Carat")
    st.subheader("We have one Lable : Price")
    X=df[["carat"]]
    y=df['price']
    X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,y, test_size=0.2)
with model_traing:
    st.header("We are train our model here:")
    model=RandomForestClassifier(n_estimators = 100) 
    model.fit(X_Train,Y_Train)
    pred=model.predict(X_Test)

    


