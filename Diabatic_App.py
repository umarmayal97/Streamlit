import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
from PIL import Image
import sklearn 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split    
from sklearn.ensemble  import RandomForestClassifier

header=st.container()
dataset=st.container()
feature=st.container()
model_traing=st.container()
plotly_1=st.container()

with header:
    st.title("Diabetes Prediction App")
    
with dataset:
    df = pd.read_csv('diabetes.csv')
    X=df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age',]]
    y=df['Outcome']
with feature:
    st.sidebar.header(' Input Patient Parameters')
    def user_input_features1():
        Pregnancies = st.sidebar.text_input('Pregnancies')
        Glucose = st.sidebar.text_input('Glucose')
        BloodPressure = st.sidebar.text_input('BloodPressure')
        SkinThickness = st.sidebar.text_input('SkinThickness')
        Insulin = st.sidebar.text_input('Insulin')
        BMI = st.sidebar.text_input('BMI')
        DiabetesPedigreeFunction = st.sidebar.text_input('Diabetes_Pedigree_Function')
        Age = st.sidebar.text_input('Age')    
        data = {'Pregnancies': Pregnancies,
                'Glucose': Glucose,
                'BloodPressure': BloodPressure,
                'SkinThickness': SkinThickness,
                'Insulin': Insulin,
                'BMI': BMI,
                'Diabetes_Pedigree_Function': DiabetesPedigreeFunction,
                'Age': Age
                }
        features1 = pd.DataFrame(data, index=[0])
        return features1

    df1 = user_input_features1()
    st.subheader('User Input parameters')
    st.write(df1)
with model_traing:
    clf = RandomForestClassifier()
    clf.fit(X, y)
    prediction = clf.predict(df1)
    try:
        st.subheader('Prediction')
        st.write(prediction)
        prediction_proba = clf.predict_proba(df1)
        if prediction[0]==0:
            color="Green"
        else:
            color="Red"
    except:
        st.text("please Enter the parameters, Thanks")
    st.subheader('Class labels and their corresponding index number')
    st.write(X.columns)
    
    st.header("Pregnancies count graph")
    st.subheader("Age vs Pregnancy")
    fig_pred=plt.figure()
    ax1=sns.scatterplot(x="Age",y="Pregnancies",data=df,hue="Outcome",
                        palette="Greens")
    ax2=sns.scatterplot(x=df1["Age"], y=df1["Pregnancies"],s=150, 
                           color="Blue")
    plt.xticks(np.arange(0,100,5))
    plt.yticks(np.arange(0,20,2))
    plt.title("Diabetes Prediction")
    st.pyplot(fig_pred)

    X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,y, test_size=0.2)
    clf.fit(X_Train, Y_Train)
    pred_y=clf.predict(X_Test)
    st.header("Score")
    accuracy=accuracy_score(Y_Test,pred_y)
    st.write("Accuracy Score : ", accuracy)
    ps=precision_score(Y_Test,pred_y)
    st.write("Precsion Score : ", ps)
    rs=recall_score(Y_Test,pred_y)
    st.write("Recall Score : ", rs)
    fs=f1_score(Y_Test,pred_y)
    st.write("F1 Score : ", rs)
    st.header("Confusion Matrix")
    cm=confusion_matrix(Y_Test,pred_y)
    st.write(cm)
    