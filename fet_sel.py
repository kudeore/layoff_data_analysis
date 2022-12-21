import numpy as np
import pandas as pd
#from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import transformers
from transformers import BertTokenizer, BertForMaskedLM, AutoModelForMaskedLM
import torch
from sklearn.ensemble import RandomForestRegressor
# from encoder import CATENCODE, PIPELINE

def feature_sel(X,Y,threshold, verbose=False):
    model =RandomForestRegressor(n_estimators=340)
    
    model.fit(X,Y)
    importances= model.feature_importances_
    final_df = pd.DataFrame({'Features':X.columns, 'Importances':importances})
    final_df.set_index('Importances')
    final_df = final_df.sort_values('Importances')
    if verbose:
        fig= px.bar(x=final_df['Features'],y=final_df['Importances'])
        fig.show()
    features = final_df['Features'][final_df['Importances']>threshold]
    return features