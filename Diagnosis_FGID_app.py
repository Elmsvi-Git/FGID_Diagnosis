# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 16:54:22 2023

@author: ElaheMsvi: el.msvi@gmail.com
"""

from Module_Diagnosis import *
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os 

#if os.path.exists('./dataset.csv'): 
#    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://magazine.jhsph.edu/sites/default/files/styles/feature_image_og_image/public/GutBrain_3200x1600.jpg?itok=RqUR2Y2C")
    st.title("Auto Diagnosis")
    choice = st.radio("Navigation", ["Upload","Profiling","Diagnosis", "Download"])
    st.info("This application helps you to classify patients based on GI symptoms")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset" , , type ={"csv"})
    if file: 
        df = pd.read_csv(file, index_col=None)
#        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Input the Patient Symptoms")
#    profile_df = df.profile_report()
#    st_profile_report(profile_df)
    Options1 = ['Never' , 'Less than one day a month' , 'One day a month' , 'Two to three days a month' , 'Once a week' , 'Two to three days a week' ,'Most days','Every day','Multiple times per day or all the time']
    R1 = st.selectbox('In the last 3 months, how often did you have a feeling of a lump or something stuck in your throat?', Options1)
    R5 = st.selectbox('In the last 3 months, how often did you have pain in the middle of your chest (not related to heart problems)?', Options1)
    row = np.array([R1 , R5]) 
    df = pd.DataFrame([row], columns = ['R1' , 'R5'])


if st.button("Click here to predict"): 
    clf = FGID_Diagnosis('model1.pkl')
    clf.load_and_clean_data(df)
    pred_test = clf.predicted_outputs()
    st.balloons()
    st.write(pred_test)
    st.success('The output is as follows\n (The last cloumns indicate the most probable clusters for each case) ')

