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
def Input_Output():
    data = st.file_uploader('Upload File', type ={"csv"})
    if data is not None:
        df = pd.read_csv(data)
        # st.write(df)
        clf = FGID_Diagnosis('model1.pkl')
        clf.load_and_clean_data(df)
        
    pred_test = ""
    if st.button("Click here to predict"):
        pred_test = clf.predicted_outputs()
        st.balloons()
    st.write(pred_test)
    st.success('The output is as follows\n (The last cloumns indicate the most probable clusters for each case) ')

if __name__== '__main__':
    Input_Output()
