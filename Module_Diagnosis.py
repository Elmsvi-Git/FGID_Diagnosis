# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 08:46:14 2023

@author: ElaheMsvi
"""

###################################################################################
# import all libraries needed
import numpy as np
import pandas as pd
import pickle
class FGID_Diagnosis():
      
        def __init__(self, model_file):
            self.clf = pickle.load(open(model_file,'rb'))
            self.data = None
###################################################################################       
        # take a data file (*.csv) and preprocess it in the same way as in the lectures
        def load_and_clean_data(self, df):
            
            # import the data
            # df = pd.read_csv(data_file,delimiter=',')
            # store the data in a new variable for later use
            self.df_with_predictions = df.copy()
            self.data = df.copy()

        # a function which outputs the probability of a data point to be 1
        def predicted_probability(self):
            if (self.data is not None):  
                pred = self.clf.predict_proba(self.data)#[:,1]
                return pred
###################################################################################        
        # a function which outputs 0 or 1 based on our model
        def predicted_output_category(self):
            if (self.data is not None):
                pred_outputs = self.clf.predict(self.data)
                return pred_outputs
###################################################################################        
        # predict the outputs and the probabilities and 
        # add columns with these values at the end of the new data
        def predicted_outputs(self):
            if (self.data is not None):
                pred = self.predicted_probability()
                most_prob_classes = np.zeros([self.data.shape[0], 4])
                most_prob_classes[: , [0,2]] = np.argsort(pred , axis = 1)[:,[-1,-2]]+1
                most_prob_classes[: , [1,3]] = np.round(np.sort(pred, axis = 1)[:,[-1,-2]],4)
                most_prob_classes_df = pd.DataFrame(data = most_prob_classes, columns = ['Most_prob_Cluster1','Probability1','Most_prob_Cluster2','Probability2'])
                #self.prob_df_with_predictions = most_prob_classes_df.copy()
                self.prob_df_with_predictions = pd.concat([most_prob_classes_df , self.df_with_predictions] , axis = 1)
                #self.df_with_predictions['Most_prob_Cluster1'] = most_prob_classes[:,0]
                #self.df_with_predictions['Probability1'] = most_prob_classes[:,1]
                #self.df_with_predictions['Most_prob_Cluster2'] = most_prob_classes[:,2]
                #self.df_with_predictions['Probability2'] = most_prob_classes[:,3]
                return self.prob_df_with_predictions

