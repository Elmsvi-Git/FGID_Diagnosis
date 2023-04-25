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
            self.select_variable()
            self.data = self.data.fillna(0)
            self.make_anxiety_dep_score()
            self.select_variable_model()
###################################################################################        
        #make psycho var.        
        def make_anxiety_dep_score(self):
            self.data['Anxiety'] = self.data['PHQ4_1']+self.data['PHQ4_2']
            self.data['Depression'] = self.data['PHQ4_3']+self.data['PHQ4_4']
            self.df_with_predictions['Anxiety'] = self.data['Anxiety']
            self.df_with_predictions['Depression'] = self.data['Depression']
###################################################################################        
#       Input variable for Identifing healty versus patient according to our criteria    

        def select_variable(self):
            selected_feature_row= ['R'+str(ii+1) for ii in range(86)]
            selected_feature_row.extend(('PHQ4_1', 'PHQ4_2','PHQ4_3','PHQ4_4') )
            self.data = self.data[selected_feature_row]
            
###################################################################################    
#       Input variable for classifcation    
        def select_variable_model(self):
            selected_feature= [ 'R1','R3', 'R4','R5','R7','R8','R9','R12',
                                'R14','R16','R18','R19',
                                'R21','R23','R25','R32','R34','R38',
                                'R40','R41','R42','R43','R45','R46',
                                'R49','R51','R52','R53','R54','R55',
                                'R59','R61','R63','R65',
                                'R68','R69','R70','R71', 'R72','R73','R74','R75',#biliary pain
                                'R80','R83','Depression','Anxiety']
            self.data_selected = self.data[selected_feature]
       
###################################################################################        
        # a function which outputs the probability of a data point to be 1
        def predicted_probability(self):
            if (self.data_selected is not None):  
                pred = self.clf.predict_proba(self.data_selected)#[:,1]
                return pred
###################################################################################        
        # a function which outputs 0 or 1 based on our model
        def predicted_output_category(self):
            if (self.data_selected is not None):
                pred_outputs = self.clf.predict(self.data_selected)
                return pred_outputs
###################################################################################        
        # predict the outputs and the probabilities and 
        # add columns with these values at the end of the new data
        def predicted_outputs(self):
            if (self.data is not None):
                patient = self.Healty_detection()
                pred = self.predicted_probability()
                most_prob_classes = np.zeros([self.data.shape[0], 4])
                # most_prob_classes[: , [0,2]] = np.argsort(pred , axis = 1)[:,[-1,-2]]+1
                most_prob_classes[patient , 0] = np.argsort(pred , axis = 1)[patient,-1]+1
                most_prob_classes[patient , 2] = np.argsort(pred , axis = 1)[patient,-2]+1
                most_prob_classes[patient , 1] = np.round(np.sort(pred, axis = 1)[patient,-1],4)
                most_prob_classes[patient , 3] = np.round(np.sort(pred, axis = 1)[patient,-2],4)
                most_prob_classes_df1 = pd.DataFrame(data = self.cluster_map(most_prob_classes[:,0]) , columns = ['Most_Prob_Cluster_First_rank'])
                most_prob_classes_df2 = pd.DataFrame(data = self.cluster_map(most_prob_classes[:,2]), columns = ['Most_Prob_Cluster_Second_rank'])
                prob_classes_df1 = pd.DataFrame(data = most_prob_classes[:,1] , columns = ['Prob1'])
                prob_classes_df2 = pd.DataFrame(data = most_prob_classes[:,3], columns = ['Prob2'])
                self.prob_df_with_predictions = pd.concat([most_prob_classes_df1 , 
                                                           prob_classes_df1 , 
                                                           most_prob_classes_df2, 
                                                           prob_classes_df2 ,
                                                           self.df_with_predictions ] , axis = 1)
                return self.prob_df_with_predictions

###################################################################################   
        #Maping the cluster numbers to cluster names
        def cluster_map(self,cluster_numbers):
            Cluster_name_dict = {
                0:'Healty',1:'Globus' , 2:'Mild Globus',
                3:'Chest pain',4: 'Heartburn',5:'Mild heartburn',6:'Dysphagia',
                7:'Postprandial fullness', 8:'Early satiation',
                9:'Constipation', 10: 'Diarrhea',11:'Mild diarrhea',12:'Bloating',
                13:'Mild bloating', 14: 'Fecal incontinence',
                15:'Predominant Globus',16:'Predominant pain during swallowing',
                17:'Predominant chest pain',18:'Predominant heart burn',
                19:'Predominant vomiting/self-induced vomiting',
                20:'Predominant regurgitation',21:'Predominant constipation',
                22:'Predominant constipation, postprandial distress',
                23:'Predominant diarrhea',
                24:'Predominant abdominal pain and pain related symptoms',
                25:'Predominant abdominal, epigastric, and biliary pain + pain related symptoms',
                26:'Biliary Pain related symptoms',
                27:'High all upper and lower GI and depression score'}
            cluster_name_list = [Cluster_name_dict.get(cluster_numbers[i]) for i in range(len(cluster_numbers))]
            return cluster_name_list
        
        def Healty_detection(self):
            
            Data = self.data.copy()
            Data = Data.fillna(0)
            thr_val = 2
            
            Glbs = np.logical_and(Data['R1']>thr_val, Data['R2']==1)
            CP = np.logical_and(Data['R5']>thr_val, Data['R6']==1)
            HB = np.logical_and(Data['R9']>thr_val, Data['R11']==1)
            FDg = np.logical_and(Data['R12']>thr_val, Data['R13']==1)
            PDS = np.logical_and(Data['R14']>thr_val, Data['R15']==1)
            ES = np.logical_and(Data['R16']>thr_val, Data['R17']==1)
            EPS =np.logical_and(Data['R18']>thr_val, Data['R20']==1)
            N = np.logical_and(np.logical_and(Data['R21']>thr_val, Data['R22']==1),Data['R25']<thr_val+1)
            V = np.logical_and(np.logical_and(Data['R23']>thr_val, Data['R24']==1),Data['R25']<thr_val+1)
            CV = np.logical_and(np.logical_and(Data['R26']==1, Data['R27']==1), Data['R28']==1)
            CHS = np.logical_and(Data['R30']==1, Data['R31']==1)
            RS = np.logical_and(Data['R32']>thr_val, Data['R33']==1)
            B = np.logical_and(Data['R38']>thr_val, Data['R39']==1)
            IBS = np.logical_and(Data['R40']>thr_val, Data['R48']==1)
            
            HS = np.logical_and(np.logical_and(Data['R49']>thr_val, Data['R50']==1),Data['R56']==1)
            L3D =np.logical_and(Data['R51']>thr_val, Data['R56']==1)
            SD =np.logical_and(Data['R52'] >thr_val, Data['R56']==1)
            IE = np.logical_and(Data['R53']>thr_val, Data['R56']==1)
            AO = np.logical_and(Data['R54']>thr_val, Data['R56']==1)
            MM = np.logical_and(Data['R55']>thr_val, Data['R56']==1)
            
            OPC = np.logical_and(Data['R57']==1, Data['R58']==1)
            FD = np.logical_and(Data['R59']>thr_val, Data['R62']==1)
            FB = np.logical_and(Data['R65']>thr_val, Data['R66']==1)
            
            CMAP = np.logical_and(np.logical_and(Data['R47']==1, Data['R40']==1), Data['R48']==1)
            # BP = np.logical_and(Data['R68']>int(thr['R68']), Data['R75']<int(thr['R75']))
            BP = np.logical_and(Data['R68']>thr_val, Data['R75']<2)

            FI = np.logical_and(Data['R80']>thr_val, Data['R82']==1)
            AP = np.logical_and(Data['R83']>thr_val,Data['R86']==1)
            patient = Glbs|CP|HB|FDg|PDS|ES|EPS|N|V|CV|CHS|RS|B|IBS|HS|L3D|SD|IE|AO|MM|OPC|FD|FB|CMAP|BP|FI|AP
            
            return patient

