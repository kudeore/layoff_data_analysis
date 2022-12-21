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

class CATENCODE:
    def __init__(self,sentences):
        self.sentences = sentences
#         self.batch_size = batch_size
        
    def sent_embedding(self,batch_size):
        os.environ["CURL_CA_BUNDLE"]=""
        url = "./bert_base_uncased/"
        tokenizer = BertTokenizer.from_pretrained(url, use_fast=True)
        model = transformers.BertModel.from_pretrained(url,
                                          output_hidden_states = True)
        model.eval()
        sequence=[]
        for idx in range(0, len(self.sentences), batch_size):
            batch = self.sentences[idx : min(len(self.sentences), idx+batch_size)]

            encoded = tokenizer.batch_encode_plus(batch,max_length=4, padding='max_length', truncation=True,return_tensors='pt')
            encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}

            with torch.no_grad():
                outputs = model(**encoded)
                hidden_states = outputs.last_hidden_state
                flattened= torch.flatten(hidden_states, start_dim=1)
                sequence.append(flattened.numpy())
        return sequence
    
    def scaling(self, data):
        sc = StandardScaler()
        sc.fit(data)
        X_sc = sc.transform(data)
        return (X_sc, sc)
    def pc(self, X,n_comp):
        pca = PCA(n_components=n_comp)
        pca.fit(X)
        X_pca = pca.transform(X)
        return (X_pca, pca)

class PIPELINE:
    def __init__(self,batch_size):
        self.batch_size = batch_size
        
    def cat_encode_pipeline(self,df_train, df_test):
        cat_features = df_train.select_dtypes(exclude=["number","bool_"]).columns
        df_ = df_train.dropna()
        df1= df_.copy()
        for i in cat_features:
            ce= CATENCODE(df1[str(i)])
            seq=ce.sent_embedding(self.batch_size)
#             sequence= np.concatenate(np.array(seq)).reshape(len(df1),int(4*768))
            sequence = np.stack(seq, axis=0)
            sequence = sequence.reshape(len(df1),int(4*768))

            #scaling and cerating pca for training
            X_sc,sc = ce.scaling(sequence)
            X_pca,pca = ce.pc(X_sc,4)

            # word embedding test data
            ce1= CATENCODE(df_test[str(i)])
            seq1=ce1.sent_embedding(self.batch_size)
#             sequence1= np.concatenate(np.array(seq1)).reshape(len(df_test),int(4*768))
            sequence1 = np.stack(seq1, axis=0)
            sequence1 = sequence1.reshape(len(df_test),int(4*768))

             #scaling and cerating pca for test data
            X_sc_test = sc.transform(sequence1)
            X_pca_test = pca.transform(X_sc_test)

            df1.drop(columns=[str(i)],inplace=True)
            col1= str(i)+'_1'
            col2= str(i)+'_2'
            df1[col1] = pd.DataFrame(X_pca)[0]
            df1[col2] = pd.DataFrame(X_pca)[1]

            df_test.drop(columns=[str(i)],inplace=True)
            col1= str(i)+'_1'
            col2= str(i)+'_2'
            df_test[col1] = pd.DataFrame(X_pca)[0]
            df_test[col2] = pd.DataFrame(X_pca)[1]


        return df1, df_test