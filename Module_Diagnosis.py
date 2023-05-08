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
import streamlit as st
from mycolorpy import colorlist as mcp
from matplotlib import pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers

class FGID_Diagnosis():
      
        def __init__(self, model_file , sc_file , data_clusters):
            self.data_train_clusters = np.loadtxt(data_clusters)    
            self.sc_clusters = pickle.load(open(sc_file,'rb'))
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
                self.pred_outputs = self.clf.predict(self.data_selected)
                return self.pred_outputs
###################################################################################        
        # predict the outputs and the probabilities and 
        # add columns with these values at the end of the new data
        def predicted_outputs(self):
            if (self.data is not None):
                self.patient = self.Healty_detection()
                self.pred = self.predicted_probability()
                most_prob_classes = np.zeros([self.data.shape[0], 4])
                # most_prob_classes[: , [0,2]] = np.argsort(pred , axis = 1)[:,[-1,-2]]+1
                most_prob_classes[self.patient , 0] = np.argsort(self.pred , axis = 1)[self.patient,-1]+1
                most_prob_classes[self.patient , 2] = np.argsort(self.pred , axis = 1)[self.patient,-2]+1
                most_prob_classes[self.patient , 1] = np.round(np.sort(self.pred, axis = 1)[self.patient,-1],4)
                most_prob_classes[self.patient , 3] = np.round(np.sort(self.pred, axis = 1)[self.patient,-2],4)
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
        
        def plot_sample(self):
            no_clusters = 27
            color1=mcp.gen_color(cmap="hsv",n=no_clusters+1)#'nipy_spectral', 'gist_ncar'
            if int(self.patient):
                test_label = np.argmax(self.pred)+1
                test_transformed = self.sc_clusters.transform(self.data_selected)
                c = color1[test_label]
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                
                fig = go.Figure()    
                fig.add_trace(go.Scatterpolar(
                    r= self.data_train_clusters[0 , :],
                    theta= self.data_selected.keys(),fill= 'toself',
                    name= 'Overall mean'))
    
                fig.add_trace(go.Scatterpolar(
                    r=self.data_train_clusters[test_label, :], 
                    theta= self.data_selected.keys(),
                    fill= 'toself',line = dict(color=c),
                    name= self.cluster_map(list([test_label]))[0]))
                
                fig.add_trace(go.Scatterpolar(
                    r= np.mean(test_transformed, axis = 0),
                    theta= self.data_selected.keys(),
                    fill= 'toself',line = dict(color='red'),
                    name= 'Patient' ))  
    
                fig.update_layout(
                    font_size = 13,showlegend = True,polar = dict(
                    bgcolor = "rgb(233, 233, 233)",angularaxis = dict(
                    linewidth = 2,showline=True,linecolor='black'),
                    radialaxis = dict(side = "counterclockwise",showline = True,
                    linewidth = 2,gridcolor = "white",gridwidth = 2,range=[-1, 2])),
                    )
                fig.show(renderer="png")  
                plt.tight_layout()
                return fig
        
        
def options_map_radio(option):
    options_dict = {'Never':0 , 'Less than one day a month':1 , 
                    'One day a month':2 , 'Two to three days a month':3 , 
                    'Once a week':4 , 'Two to three days a week':5 ,
                    'Most days':6,'Every day':7,
                    'Multiple times per day or all the time':8}
    options_number = options_dict.get(option)
            
    return options_number

def options_map_slider(option):
    options_dict = {0:0 , 10:1, 20:2, 30:3, 40:4 , 50:5,
                    60:6, 70:7, 80:8, 90:9, 100:10}
    options_number = options_dict.get(option)

    return options_number

def options_map_box(option):
    options_dict = {'No':0 , 'Yes':1} 
    options_number = options_dict.get(option)
    return options_number

def options_map_box_HB(option):
    options_dict = {'No':0 , 'Yes':1 ,
                    'I dont know because I have not stopped using cannabis that long':2 } 
    options_number = options_dict.get(option)
    return options_number

def options_map_box_CB(option):
    options_dict = {'Never':0 , 'casionally':1 ,
                    'Regularly':2 } 
    options_number = options_dict.get(option)
    return options_number


def options_map_PHQ4(option):
    options_dict = {'Not at all':0,'Several days':1,
                    'More than half the days':2,'Nearly every day':3}

    options_number = options_dict.get(option)
    return options_number

def get_profile():
    Options1 = ['Never' , 'Less than one day a month' , 'One day a month' , 'Two to three days a month' , 'Once a week' , 'Two to three days a week' ,'Most days','Every day','Multiple times per day or all the time']
    Options_YN = ['No' , 'Yes' ]
    Options_HB = ['No','Yes' , 'I dont know because I have not stopped using cannabis that long']
    Options_Phsyc =['Not at all','Several days','More than half the days','Nearly every day']
    feature_name = ['R' + str(i+1) for i in range(86)]
    feature_name.extend(('PHQ4_1', 'PHQ4_2','PHQ4_3','PHQ4_4') )
    df_prof = pd.DataFrame(data = np.zeros([1 , 90]) , columns= feature_name)
    R1 = st.radio('**R1. In the last 3 months, how often did you have a feeling of a lump or something stuck in your throat?**', Options1)
    df_prof['R1'] = options_map_radio(R1)
    if R1 != 'Never':        
        df_prof['R2'] = options_map_box(st.selectbox('**R2. Has it been 6 months or longer since you started having this feeling of lump or something stuck in your throat?**' , Options_YN))
        df_prof['R3'] = options_map_radio(st.radio('**R3. How often did the feeling of a lump or something stuck in your throat happen between meals - when you were not eating? (Percent of times when you had this feeling)**' , Options1))
        df_prof['R4'] = options_map_slider(st.slider('**R4. When you had the feeling of a lump or something stuck in your throat, how often did it hurt to swallow? (Percent of times when you had this feeling)**', 0, 100, 0, 10))
    R5 = st.radio('**R5. In the last 3 months, how often did you have pain in the middle of your chest (not related to heart problems)?**', Options1)
    df_prof['R5'] = options_map_radio(R5)
    if R5 != 'Never':
        df_prof['R6'] = options_map_box(st.selectbox('**R6. Has it been 6 months or longer since you started having this pain in the middle of your chest?**', Options_YN))
        df_prof['R7'] = options_map_slider(st.slider('**R7. When you had this chest pain, how often did it feel like burning? (Percent of times with this chest pain)**', 0, 100, 0, 10))
        df_prof['R8'] = options_map_slider(st.slider('**R8. When you had this symptom of chest pain, how often was it associated with food sticking after swallowing? (Percent of times with this chest pain)**', 0, 100, 0, 10))
    R9 = st.radio('**R9. In the last 3 months, how often did you have heartburn (a burning discomfort or burning pain in your chest)?**', Options1)
    df_prof['R9'] = options_map_radio(R9)
    if R9!='Never':
        df_prof['R11'] = options_map_box(st.selectbox('**R11. Has it been 6 months or longer since you started having this heartburn (burning discomfort or burning pain in your chest)?**', Options_YN))
    R12 = st.radio('**R12. In the last 3 months, how often did food or drinks get stuck in your chest after swallowing or go down slowly through your chest?**' , Options1)
    df_prof['R12'] = options_map_radio(R12)
    if R12!='Never':    
        df_prof['R13'] = options_map_box(st.selectbox('**R13. Has it been 6 months or longer since you started having this problem in your chest of food getting stuck or going down slowly?**', Options_YN))
    R14 = st.radio('**R14. In the last 3 months, how often did you feel so full after a regular-sized meal (the amount you normally eat) that it interfered with your usual activities?**', Options1)
    df_prof['R14'] = options_map_radio(R14)
    if R14!='Never':       
        df_prof['R15'] = options_map_box(st.selectbox('**R15. Has it been 6 months or longer since you started having these episodes of fullness after meals that was severe enough to interfere with your usual activities?**', Options_YN))
    R16 = st.radio('**R16. In the last 3 months, how often were you unable to finish a regular-sized meal because you felt too full?**', Options1)
    df_prof['R16']  = options_map_radio(R16)
    if R16!='Never':       
        df_prof['R17'] = options_map_box(st.selectbox('**R17. Has it been 6 months or longer since you started having these episodes of feeling too full to finish regular-sized meals?**', Options_YN) )
    R18 = st.radio('**R18. In the last 3 months, how often did you have pain or burning in the middle part of your upper abdomen (above your belly button but not in your chest), that was so severe that it interfered with your usual activities?**', Options1)
    df_prof['R18'] = options_map_radio(R18)
    if R18!='Never':        
        df_prof['R19'] = options_map_slider(st.slider('**R19. When you had this pain or burning in the middle part of your upper abdomen, how often did it stop or get better after bowel movement or passing gas? (Percent of times you had the pain or burning)**', 0, 100, 0, 10))
        df_prof['R20'] = options_map_box(st.selectbox('**R20. Has it been 6 months or longer since you started having this pain or burning in the middle part of your upper abdomen?**', Options_YN))
    R21 = st.radio('**R21. In the last 3 months, how often did you have nausea that was so severe that it interfered with your usual activities?**', Options1)
    df_prof['R21'] = options_map_radio(R21)
    if R21!='Never':      
        df_prof['R22'] = options_map_box(st.selectbox('**R22. Has it been 6 months or longer since you started having this nausea?**', Options_YN))
    R23 = st.radio('**R23. In the last 3 months, how often did you vomit?**', Options1)
    df_prof['R23'] = options_map_radio(R23)
    if R23!='Never':   
        df_prof['R24'] = options_map_box(st.selectbox('**R24. Has it been 6 months or longer since you started having this vomiting?**', Options_YN))
        df_prof['R25'] = options_map_slider(st.slider('**R25. Did you make yourself vomit? (Percent of times you vomited)**' , 0, 100, 0, 10))
        df_prof['R26'] = options_map_box(st.selectbox('**R26. Did you have episodes in the last year of frequent vomiting that started suddenly, lasted up to one week, and then stopped?**', Options_YN))
        df_prof['R27'] = options_map_box(st.selectbox('**R27. Did you have at least three such episodes of frequent vomiting in the last year?**', Options_YN))
        df_prof['R28'] = options_map_box(st.selectbox('**R28. Between the episodes of frequent vomiting, were there one or more weeks when you did not vomit at all?**', Options_YN))
        df_prof['R29'] = options_map_box(st.selectbox('**R29. Did you take hot baths or showers to relieve your vomiting?**', Options_YN))
    R30 = st.selectbox('**R30. Do you use cannabis (pot or marijuana)?**' , ('Never','Occasionally','Regularly'))
    df_prof['R30'] = options_map_box_CB(R30)
    if R30!='Never':
        df_prof['R31'] = options_map_box_HB(st.selectbox('**R31. When you stopped using cannabis for several weeks, did the vomiting disappear?**' , Options_HB))
    R32 = st.radio('**R32. In the last 3 months, how often did food come back up into your mouth after you swallowed it?**', Options1)
    df_prof['R32'] = options_map_radio(R32)
    if R32 !='Never':        
        df_prof['R33'] = options_map_box(st.selectbox('**R32. Has it been 6 months or longer since you started having this problem (food coming back up into your mouth)?**', Options_YN))
        df_prof['R34'] = options_map_slider(st.slider('**R33.How often did you have retching (heaving) before food came back up into your mouth? (Percent of times when food came back up)**' , 0, 100, 0, 10))
        df_prof['R35'] = options_map_slider(st.slider('**R34.When food came back up into your mouth, how often did you vomit or feel sick to your stomach? (Percent of times when food came back up)**' , 0, 100, 0, 10))
        df_prof['R36'] = options_map_box(st.selectbox('**R35.Did the food stop coming back up into your mouth when it turned sour?**', Options1))
        df_prof['R37'] = options_map_slider(st.slider('**R36.How often was the food that came back up into your mouth recognizable food with a pleasant taste? (Percent of times when food came back up)**' , 0, 100, 0, 10))
    R38 = st.radio('**R38. In the last 3 months, how often did you experience belching that was so severe that it interfered with your usual activities?**', Options1)
    df_prof['R38'] = options_map_radio(R38)
    if R38 !='Never': 
        df_prof['R39'] = options_map_box(st.selectbox('**R39. Has it been 6 months or longer since you started having this belching that was severe enough to interfere with your usual activities?**', Options_YN))
    R40 = st.radio('**R40. In the last 3 months, how often did you have pain anywhere in your abdomen?**', Options1)
    df_prof['R40'] = options_map_radio(R40)
    if R40 !='Never': 
        df_prof['R41'] = options_map_slider(st.slider('**R41. How often did this pain in your abdomen happen close in time to a bowel movement -- just before, during, or soon after? (Percent of times with pain)**' , 0, 100, 0, 10))
        df_prof['R42'] = options_map_slider(st.slider('**R42. How often did your stools become either softer than usual or harder than usual when you had this pain? (Percent of times with pain)**' , 0, 100, 0, 10))
        df_prof['R43'] = options_map_slider(st.slider('**R43. How often did your stools become either more frequent than usual or less frequent than usual when you had this pain? (Percent of times with pain)**' , 0, 100, 0, 10))
        df_prof['R45'] = options_map_slider(st.slider('**R45. How often did your pain start or get worse after eating a meal? (Percent of times with pain)**' , 0, 100, 0, 10))
        df_prof['R46'] = options_map_slider(st.slider('**R46. When you had this pain, how often did it limit or restrict your usual activities (for example, work, household activities, and social events)? (Percent of times with pain)**' , 0, 100, 0, 10))
        df_prof['R47'] = options_map_box(st.selectbox('**R47. Has this pain in your abdomen been continuous or almost continuous? (Continuous means that it never goes away during waking hours)**', Options_YN))
        df_prof['R48'] = options_map_box(st.selectbox('**R48. Has it been 6 months or longer since you started having this pain?**', Options_YN))
    df_prof['R49'] = options_map_slider(st.slider('**R49. In the last 3 months, how often did you have hard or lumpy stools? (Percent of all bowel movements)**' , 0, 100, 0, 10))
    df_prof['R51'] = options_map_slider(st.slider('**R51. In the last 3 months, how often did you have fewer than three bowel movements a week without taking a laxative medication or enema?  (Percent of weeks)**' , 0, 100, 0, 10))
    df_prof['R52'] = options_map_slider(st.slider('**R52. In the last 3 months, how often did you strain during bowel movements? (Percent of bowel movements)**' , 0, 100, 0, 10))
    df_prof['R53'] = options_map_slider(st.slider('**R53. In the last 3 months, how often did you have a feeling of incomplete emptying after bowel movements? (Percent of bowel movements)**' , 0, 100, 0, 10))
    df_prof['R54'] = options_map_slider(st.slider('**R54. In the last 3 months, how often did you have a sensation that the stool could not be passed (was blocked), when having a bowel movement? (Percent of bowel movements)**' , 0, 100, 0, 10))
    df_prof['R55'] = options_map_slider(st.slider('**R55. In the last 3 months, how often did you press on or around your bottom, or remove stool with your fingers, in order to have a bowel movement? (Percent of bowel movements)**' , 0, 100, 0, 10))
    df_prof['R56'] = options_map_box(st.selectbox('**R56. Did any of the symptoms of constipation listed in the 6 previous questions begin more than 6 months ago?**', Options_YN))
    R57 = st.selectbox('**R57. Are you currently taking a prescription medication for pain?**', Options_YN)#--->>>> 58!!!!!!
    df_prof['R57'] = options_map_box(R57)
    if R57 !='No': 
        df_prof['R58'] = options_map_box(st.selectbox('**R58. Have any of the constipation symptoms listed in questions R49-R55 above changed since you started taking prescription medication for pain?**', Options_YN))
    df_prof['R59'] = options_map_slider(st.slider('**R59. In the last 3 months, how often did you have mushy or watery stools when you were not using drugs or other treatment for constipation? (Percent of all bowel movements)**' , 0, 100, 0, 10))
    df_prof['R60'] = options_map_box(st.selectbox('**R60. Did you have mushy and watery stools when you were not using drugs or other treatment for constipation?**', Options_YN))
    df_prof['R61'] = options_map_slider(st.slider('**R61. How often did your symptom of mushy or watery stools happen following a meal? (Percent of mushy or liquid stools)**' , 0, 100, 0, 10))
    df_prof['R62'] = options_map_box(st.selectbox('**R62. Has it been 6 months or longer since you started having frequent mushy or watery stools?**', Options_YN))
    df_prof['R63'] = options_map_slider(st.slider('**R63. In the last 3 months, how often did you have to rush to the toilet to have a bowel movement? (Percent of bowel movements)**' , 0, 100, 0, 10))
    R65 = st.slider('**R65. In the last 3 months, how often did you feel bloated or notice that your belly looked unusually large?**', 0, 100, 0, 10)
    df_prof['R65'] = options_map_slider(R65)
    if R65 !='Never': 
        df_prof['R66'] = options_map_box(st.selectbox('**R66. Has it been 6 months or longer since you started having this problem of feeling bloated or your belly looking unusually large?**', Options_YN))
    R68 = st.selectbox('**R68. In the last 6 months, how often did you have pain in the middle or right side of your upper abdomen?**', Options_YN)
    df_prof['R68'] = options_map_box(R68)
    if R68!='Never':
        df_prof['R69'] = options_map_slider(st.slider('**R69. How often did this pain last 30 minutes or longer? (Percent of times with pain)**' , 0, 100, 0, 10))
        df_prof['R70'] = options_map_slider(st.slider('**R70. How often did this pain build up to a steady, severe level? (Percent of times with pain)**' , 0, 100, 0, 10))
        df_prof['R71'] = options_map_slider(st.slider('**R71. How often did this pain go away completely between episodes? (Percent of times with pain)**' , 0, 100, 0, 10))
        df_prof['R72'] = options_map_slider(st.slider('**R72. How often did this pain stop you from your usual activities, or cause you to see a doctor urgently or go to the emergency department? (Percent of times with pain)**' , 0, 100, 0, 10))
        df_prof['R73'] = options_map_slider(st.slider('**R73. How often did this pain in your upper abdomen happen just before, during, or immediately following a bowel movement? (Percent of times with pain)**' , 0, 100, 0, 10))
        df_prof['R74'] = options_map_slider(st.slider('**R74. How often was this pain improved by changing your position from lying down to sitting, or from sitting to standing? (Percent of times with pain)**' , 0, 100, 0, 10))
        df_prof['R75'] = options_map_slider(st.slider('**R75. How often was this pain improved by taking medication to reduce acid? (Percent of times with pain)**' , 0, 100, 0, 10))
    R80 = st.radio('**R80. In the last 3 months, how often did you have accidental leakage of stool (fecal material)?**',Options1)
    df_prof['R80'] = options_map_radio(R80)
    if R80!='Never':
        df_prof['R81'] = options_map_radio(st.radio('**R81. In the last 3 months, when this leakage occurred, on average what was the amount of stool that leaked?**',Options1))
        df_prof['R82'] = options_map_box(st.selectbox('**R82. Has it been 6 months or longer since you started having accidental leakage of stool?**', Options_YN))
    R83 = st.radio('**R83. In the last 3 months, how often have you had aching, pain, or pressure in the rectum when you were not having a bowel movement? (The rectum is the portion of your colon or large bowel just above the anal opening).**',Options1)
    df_prof['R83'] = options_map_radio(R83)
    if R83!='Never':
        df_prof['R86'] = options_map_box(st.selectbox('**R86. Has it been 6 months or longer since you started having this aching, pain or pressure in the rectum?**', Options_YN))
    df_prof['PHQ4_1'] = options_map_PHQ4(st.radio('**Over the last 2 weeks, how often have you been bothered by feeling nervous, anxious or on edge**' , Options_Phsyc))
    df_prof['PHQ4_2'] = options_map_PHQ4(st.radio('**Over the last 2 weeks, how often have you been bothered by not being able to stop or control worrying**', Options_Phsyc))
    df_prof['PHQ4_3'] = options_map_PHQ4(st.radio('**Over the last 2 weeks, how often have you been bothered by little interest or pleasure in doing things**',Options_Phsyc))
    df_prof['PHQ4_4'] = options_map_PHQ4(st.radio('**Over the last 2 weeks, how often have you been bothered by feeling down, depressed, or hopeless**',Options_Phsyc))

    return df_prof
    
