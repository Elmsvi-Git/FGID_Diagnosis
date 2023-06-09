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

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')


with st.sidebar: 
    st.image("https://magazine.jhsph.edu/sites/default/files/styles/feature_image_og_image/public/GutBrain_3200x1600.jpg?itok=RqUR2Y2C")
    st.title("Gut-Brain Disorders Diagnosis")
    choice = st.radio("**How do you want to input the patient symptoms?**", ["Upload","Profiling"])
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
        
        # if file is not None:
        #     file_details = {"FileName":file.name,"FileType":file.type}
        #     st.write(file_details)
        #     with open(os.path.join("tempDir",file.name),"wb") as f: 
        #       f.write(file.getbuffer())         
        #     st.success("Saved File")

if choice == "Profiling": 
    st.title("Input the Patient Symptoms")

#    profile_df = df.profile_report()
#    st_profile_report(profile_df)
    df_prof = get_profile()
    df = df_prof.copy()
    
if st.button("Click here to predict"): 
    clf = FGID_Diagnosis('model2.pkl' , 'sc_train.pkl','Data_Clusters.txt')
    clf.load_and_clean_data(df)
    pred_test = clf.predicted_outputs()
    st.success('The output is as follows: ')
    # st.dataframe(pred_test)
    st.write(pred_test)
    if df.shape[0]==1:
        fig_one_smaple = clf.plot_sample()
        if fig_one_smaple:
            st.plotly_chart(fig_one_smaple, use_container_width=True)

    csv = convert_df(pred_test)
    st.download_button(label="Download output file as .csv",data=csv, file_name='output.csv',mime='text/csv',)

if st.button("Learn about the clusters"): 
    image = Image.open("Clusters.jpg")
    st.image(image, caption='Complex Clusters')


