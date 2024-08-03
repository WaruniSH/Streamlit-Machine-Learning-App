import time
import streamlit as st
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
import pickle
from sklearn.preprocessing import MinMaxScaler

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key: 
            return float(value)

app_mode = st.sidebar.selectbox('Select Page',['Home','Data Visualization','Data Prediction'])

def predictCancer(data):

    #import scaler
    with open('scaler.pkl', 'rb') as file:
        scaler1 = pickle.load(file)
    df2 = pd.DataFrame(data)
    record_to_scale = df2.iloc[0:1]
    num_vars=['AGE']
    df2[num_vars]=scaler1.transform(record_to_scale[num_vars])

    single_sample = np.array(df2).reshape(1, -1)

    #import model
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    prediction= model.predict(single_sample)

    return prediction


if app_mode =='Home':

    st.title ("LungGuard Cancer Prediction App !")

    st.image("lung_cancer.png")

    container1=st.container()
    container1.subheader("What is Lung Cancer?")
    container1.markdown("Lung cancer is one of the most common and serious types of cancer. It starts in the lungs and can spread to other parts of the body. Early detection is crucial as it significantly increases the chances of successful treatment.")

    container2=st.container()
    container2.subheader("Types of Lung Cancer")
    container2.markdown("1. Non-Small Cell Lung Cancer (NSCLC): The most common type, accounting for about 85% of cases.")
    container2.markdown("2. Small Cell Lung Cancer (SCLC): A less common but more aggressive form of lung cancer.")

    container3=st.container()
    container3.subheader("What cause for Lung Cancer")
    container3.markdown("1. Smoking: The leading cause of lung cancer, responsible for about 85% of cases.")
    container3.markdown("2. Exposure to Radon Gas: A naturally occurring gas that can accumulate in buildings..")
    container3.markdown("3. Asbestos Exposure: Often occurs in certain work environments.")
    container3.markdown("4. Family History: Genetic factors can also play a role.")

    container4=st.container()
    container4.subheader("How APP work?")
    container4.markdown("App leverages advanced artificial intelligence and machine learning to predict the risk of lung cancer. Our technology analyzes a wide range of health data to provide accurate and personalized insights.")

elif app_mode=='Data Visualization':

    st.title('Data Visualization')
    st.header('Lung Cancer Prediction Dataset')
    data=pd.read_csv('survey_lung_cancer.csv')
    st.write(data.head())

    st.header('Bar charts')
    st.subheader('Gender Distribution')
    st.bar_chart(data['GENDER'  ].head(20))

    st.header("line chart")
    st.subheader('Chest pain vs Smoking ')
    df= pd.DataFrame(
                np.random.randn(10, 2),
                        columns=['SMOKING', 'CHEST PAIN'])
    st.line_chart(df)


    st.subheader('Anxiety vs Chest pain ')
    df= pd.DataFrame(
                np.random.randn(10, 2),
                        columns=['ANXIETY', 'CHEST PAIN'])
    st.line_chart(df)

    #Smoking distributuion
    st.subheader('Smoking Distribution')
    fig, ax = plt.subplots()
    sns.histplot(data['SMOKING'],  bins=4,ax=ax)
    ax.set_title('Smoking Distribution of Dataset')
    ax.set_xlabel('Smoking')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    #Lung Cancer distributuion
    st.subheader('Lung Cancer Distribution')
    fig, ax = plt.subplots()
    sns.histplot(data['LUNG_CANCER'],  bins=15, ax=ax)
    ax.set_title('Lung Cancer of Dataset')
    ax.set_xlabel('Lung Cancer')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)


    #Alchohol distributuion
    st.subheader('Alchohol Consumption')
    fig, ax = plt.subplots()
    sns.histplot(data['ALCOHOL CONSUMING'], bins=5, ax=ax)
    ax.set_title('Alchohol of Dataset')
    ax.set_xlabel('Alchohol')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    #Age boxplot
    st.subheader('Age Distribution Box Plot ')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data['AGE'])
    ax.set_xlabel('Age')
    ax.set_ylabel('Value')
    st.pyplot(fig)

    #Lung Cancer Distribution by Gender
    st.subheader('Lung Cancer Distribution by Gender ')
    gender_lung_cancer_counts = data.groupby(['GENDER', 'LUNG_CANCER']).size().reset_index(name='COUNT')
    bar_chart = alt.Chart(gender_lung_cancer_counts).mark_bar().encode(
        x=alt.X('GENDER', title='Gender'),
        y=alt.Y('COUNT', title='Count'),
        color='LUNG_CANCER'
    )
    st.altair_chart(bar_chart, use_container_width=True)

    #Age Distribution of Lung Cancer Patients
    st.subheader('Age Distribution of Lung Cancer Patients')
    lung_cancer_df = data[data['LUNG_CANCER'] == 'YES']
    histogram = alt.Chart(lung_cancer_df).mark_bar().encode(
        x=alt.X('AGE', bin=True, title='Age'),
        y=alt.Y('count()', title='Count')
    )
    st.altair_chart(histogram, use_container_width=True)

    #Smoking Status and Lung Cancer Correlation
    st.subheader('Smoking Status and Lung Cancer Correlation')
    smoking_lung_cancer_counts = data.groupby(['SMOKING', 'LUNG_CANCER']).size().reset_index(name='COUNT')
    bar_chart = alt.Chart(smoking_lung_cancer_counts).mark_bar().encode(
        x=alt.X('SMOKING:N', title='Smoking Status'),
        y=alt.Y('COUNT:Q', title='Count'),
        color='LUNG_CANCER:N'
    )
    st.altair_chart(bar_chart, use_container_width=True)
    
    #Chronic Disease and Lung Cancer Correlation
    st.subheader('Chronic Disease and Lung Cancer Correlation')
    chronic_disease_lung_cancer_counts = data.groupby(['CHRONIC DISEASE', 'LUNG_CANCER']).size().reset_index(name='COUNT')
    bar_chart = alt.Chart(chronic_disease_lung_cancer_counts).mark_bar().encode(
        x=alt.X('CHRONIC DISEASE:N', title='Chronic Disease'),
        y=alt.Y('COUNT:Q', title='Count'),
        color='LUNG_CANCER:N'
    )
    st.altair_chart(bar_chart, use_container_width=True)

    #heatmap
    st.subheader('Correlation Heatmap')
    # Convert categorical variables to numerical codes 
    data['GENDER'] = data['GENDER'].map({'M': 0, 'F': 1})
    data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'NO': 0, 'YES': 1})

    #correlation matrix
    corr = data.corr()
    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
    # Display the heatmap 
    st.pyplot(plt)
 

elif app_mode=='Data Prediction':
    st.title('Lung Cancer Prediction')
    gender_dict = {"Male": 1, "Female": 0}
    Yes_No_dict = {"Yes": 2, "No": 1}
  
    # getting the input data from the user
    Gender = st.radio('Gender', tuple(gender_dict.keys()))
    Age1 = st.number_input('Age of the Person',value=None)
    Age = 0.0
    if Age1 is not None:
        Age = float(Age1)

    Smoking = st.radio('Do you smoking', tuple(Yes_No_dict.keys()))
    Yellow_Fingers = st.radio('Do you have Yellow_Fingers', tuple(Yes_No_dict.keys()))
    Anxiety = st.radio('Do you feel Anxiety', tuple(Yes_No_dict.keys()))
    Peer_Pressure = st.radio('Do you have Peer_Pressure', tuple(Yes_No_dict.keys()))
    Chronic_Disease = st.radio('Do you have Chronic_Disease', tuple(Yes_No_dict.keys()))
    Fatigue = st.radio('Do you have Fatigue', tuple(Yes_No_dict.keys()))
    Allergy = st.radio('Do you have Allergy', tuple(Yes_No_dict.keys()))
    Wheezing = st.radio('Do you have Wheezing', tuple(Yes_No_dict.keys()))
    Alcohol_Consuming = st.radio('Do you consume Alcohol', tuple(Yes_No_dict.keys()))
    Coughing = st.radio('Do you have Coughing', tuple(Yes_No_dict.keys()))
    Shortness_of_Breath = st.radio('Do you have Shortness_of_Breath', tuple(Yes_No_dict.keys()))
    Swallowing_Difficulty = st.radio('Do you have Swallowing_Difficulty', tuple(Yes_No_dict.keys()))
    Chest_Pain = st.radio('Do you have Chest_Pain', tuple(Yes_No_dict.keys()))

    data = {
        'GENDER': [get_value(Gender, gender_dict)],
        'AGE': [Age],
        'SMOKING': [get_value(Smoking, Yes_No_dict)],
        'YELLOW_FINGERS': [get_value(Yellow_Fingers, Yes_No_dict)],
        'ANXIETY': [get_value(Anxiety, Yes_No_dict)],
        'PEER_PRESSURE': [get_value(Peer_Pressure, Yes_No_dict)],
        'CHRONIC DISEASE': [get_value(Chronic_Disease, Yes_No_dict)],
        'FATIGUE': [get_value(Fatigue, Yes_No_dict)],
        'ALLERGY': [get_value(Allergy, Yes_No_dict)],
        'WHEEZING': [get_value(Wheezing, Yes_No_dict)],
        'ALCOHOL CONSUMING': [get_value(Alcohol_Consuming, Yes_No_dict)],
        'COUGHING': [get_value(Coughing, Yes_No_dict)],
        'SHORTNESS OF BREATH': [get_value(Shortness_of_Breath, Yes_No_dict)],
        'SWALLOWING DIFFICULTY': [get_value(Swallowing_Difficulty, Yes_No_dict)],
        'CHEST PAIN': [get_value(Chest_Pain, Yes_No_dict)]
        }

    if st.button("Cancer Predict"):
        with st.spinner('Loading Results...'):
            time.sleep(4)
            result=predictCancer(data)
        
            if(result==1):
                st.error('This person has Lung Cancer')
            else:
                st.success('This person does not have Lung Cancer')

   


