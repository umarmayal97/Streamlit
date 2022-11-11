import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import sklearn 
from sklearn import datasets
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split    
from sklearn.ensemble  import RandomForestClassifier

header=st.container()
dataset=st.container()
feature=st.container()
model_traing=st.container()
plotly_1=st.container()

with header:
    st.header("Flower App")
    
with dataset:
    st.header("Load DataSet")
    df = sns.load_dataset('iris')
    X=df[['sepal_length', 'sepal_width', 'petal_length','petal_width']]
    y=df['species']
with feature:
    st.sidebar.header('User Input Parameters')
    def user_input_features1():
        sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
        sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
        petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
        data = {'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width}
        features1 = pd.DataFrame(data, index=[0])
        return features1

    df1 = user_input_features1()
    st.subheader('User Input parameters')
    st.write(df1)
with model_traing:
    clf = RandomForestClassifier()
    clf.fit(X, y)
    prediction = clf.predict(df1)
    prediction_proba = clf.predict_proba(df1)
    st.subheader('Class labels and their corresponding index number')
    st.write(X.columns)
    st.subheader('Prediction')
    #st.write(y.columns)
    st.write(prediction)
    
with plotly_1:
    #Flower_name = df['species'].unique().tolist()
   # year=st.selectbox("which flower you want", Flower_name,0)
    fig=px.scatter(df,x="sepal_length",y='sepal_width',color='petal_length', size_max=55,
                    range_x=[0,10],range_y=[0,10],animation_frame='species')
    
    st.write(fig)