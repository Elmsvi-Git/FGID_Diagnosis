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
from PIL import Image

#if os.path.exists('./dataset.csv'): 
#    df = pd.read_csv('dataset.csv', index_col=None)

# @st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

with st.sidebar: 
    st.image("https://magazine.jhsph.edu/sites/default/files/styles/feature_image_og_image/public/GutBrain_3200x1600.jpg?itok=RqUR2Y2C")
    st.title("Gut-Brain Disorders Diagnosis")
    choice = st.radio("Navigation", ["Upload","Profiling"])
    st.info("This application helps you to classify patients based on gastrointestinal and psychological symptoms")

if choice == "Upload":
    st.title("Upload Your Dataset")
    
    st.info("You can download a sample input file from here:")
    input_file = pd.read_csv('sample_file.csv' , index_col=None)
    csv = convert_df(input_file)
    st.download_button(label="Download data as CSV", data=csv,
                       file_name='sample_input.csv',mime='text/csv' )
    
    file = st.file_uploader(" " , type ={"csv"})
    if file: 
        df = pd.read_csv(file, index_col=None)
#        df.to_csv('dataset.csv', index=None)
        #st.dataframe(df)

if choice == "Profiling": 
    st.title("Input the Patient Symptoms")
    st.info("This Section is not complete yet....")

#    profile_df = df.profile_report()
#    st_profile_report(profile_df)
    Options1 = ['Never' , 'Less than one day a month' , 'One day a month' , 'Two to three days a month' , 'Once a week' , 'Two to three days a week' ,'Most days','Every day','Multiple times per day or all the time']
    R1 = st.selectbox('In the last 3 months, how often did you have a feeling of a lump or something stuck in your throat?', Options1)
    R5 = st.selectbox('In the last 3 months, how often did you have pain in the middle of your chest (not related to heart problems)?', Options1)
    row = np.array([R1 , R5]) 
    df = pd.DataFrame([row], columns = ['R1' , 'R5'])


if st.button("Click here to predict"): 
    clf = FGID_Diagnosis('model2.pkl')
    clf.load_and_clean_data(df)
    pred_test = clf.predicted_outputs()
    st.success('The output is as follows: ')
    #st.balloons()
    st.dataframe(pred_test)
    # st.write(pred_test)

if st.button("Download output file"): 
    st.success('Maybe later ...')    
    # csv_out = convert_df(pred_test)
    # pred_test.to_csv('output.csv', index=None)
    # st.download_button(label="Download out out data as CSV", data=csv_out,
    #                    file_name='output_file.csv',mime='text/csv' )

if st.button("Learn about the clusters"): 
    image = Image.open("Clusters.jpg")
    st.image(image, caption='Complex Clusters')




    
  
