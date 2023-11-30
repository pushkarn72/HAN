import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os
# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import Model
# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input
import pickle 
from pickle import dump
import shutil
# import transformers as ppb
import torch
from itertools import accumulate
# import math
# from math import ceil


 
 

sheets_dict = pd.read_csv("data/csvdata/otherlanguageTest2.csv")
 




with open("tokenizer_bert.pickle","rb") as handle:
    tokenizer2=pickle.load(handle)
    
    
vector={}

cnt=0


max_sent_len=25   # 16 earlier
max_word_len=10   # 


label,senti,emoti,severity=[],[],[],[]

for i in range(len(sheets_dict)):

    #label.append(sheets_dict["Bully_Label"][i])
    #senti.append(sheets_dict["Sentiment_label"][i])
    #emoti.append(sheets_dict["Emotion_label"][i])
    #sar.append(sheets_dict["Sarcasm"][i])
    label.append(sheets_dict["Complain Label"][i])
    senti.append(sheets_dict["Sentiment"][i])
    emoti.append(sheets_dict["Emotion"][i])
    severity.append(sheets_dict["Severity"][i])






for i in range(len(sheets_dict)):

    doc=[]
    sent_per_doc=[]
    word_per_sent=[]

    print("step : ",i)
    #print(sheets_dict["Bully_Label"][i])
             
    #sentences=[str(sheets_dict["Processed_Tweets"][i])]
    sentences =[str(sheets_dict["review"][i])]
    
        
    
    #print(sentences)
    

    
    #embedding=get_feature(sentences)
    max_len=128
    embedding=tokenizer2.texts_to_sequences(sentences)[0] ### cover Text to sequnce number
    
    
    #sent_per_doc.append(ceil(len(embedding)/4))
    sent_per_doc.append(16)
   
    
   # for i in range(max_sent_len):
   #     if(len(embedding)%4==0):
   #         if(i<ceil(len(embedding)/4)):
   #             word_per_sent.append(4)
   #         else:
   #             word_per_sent.append(1)
   #     else:
   #         if(i<ceil(len(embedding)/4)-1):
   #             word_per_sent.append(4)
   #         elif(i==(ceil(len(embedding)/4)-1)):
   #             word_per_sent.append(len(embedding)%4)
   #         else:
   #             word_per_sent.append(1)
    
    
    for i in range(max_sent_len):
        word_per_sent.append(4)
    
    
    for i in range(100-len(embedding)):
        embedding.append(0)
        
    split_len=[]
        
    for i in range(max_sent_len):
        split_len.append(4)
    
    doc.append([embedding[x - y: x] for x, y in zip(accumulate(split_len), split_len)])
        
        
    
        
    
    sent_per_doc=np.array(sent_per_doc)
    word_per_sent=np.array(word_per_sent)
    doc=np.array(doc)
    
    sent_per_doc=sent_per_doc.reshape(1,-1)
    word_per_sent=word_per_sent.reshape(1,max_sent_len)
    doc=doc.reshape(1,max_sent_len,4)
    
    
    #print(sent_per_doc.shape)
    #print(word_per_sent.shape)
    #print(doc.shape)
    
 
       
    if cnt==0:
        #vector["bully-label"]=np.array(img_path)    
        
        vector["doc"]=doc
        vector["sent_per_doc"]=sent_per_doc
        vector["word_per_sent"]=word_per_sent
        #vector["emoti-label"]=np.array(sheets_dict["Emotion_label"][i])
        #vector["senti-label"]=np.array(sheets_dict["Sentiment_label"][i])
        #vector["sarcasm-label"]=np.array(sheets_dict["Sarcasm"][i])
        #vector["task-label"]=np.array(sheets_dict["Bully_Label"][i])
    
        print(vector["doc"].shape)
        cnt=cnt+1

  
    else:
    
        #vector["bully-label"]=np.vstack((vector["bully-label"],img_path))
        #print(vector["bully-label"].shape)

        
        vector["doc"]=np.vstack((vector["doc"],doc))
        vector["sent_per_doc"]=np.vstack((vector["sent_per_doc"],sent_per_doc))
        vector["word_per_sent"]=np.vstack((vector["word_per_sent"],word_per_sent))
        #vector["emoti-label"]=np.vstack((vector["emoti-label"],np.array(sheets_dict["Emotion_label"][i])))
        #vector["senti-label"]=np.vstack((vector["senti-label"],np.array(sheets_dict["Sentiment_label"][i])))
        #vector["sarcasm-label"]=np.vstack((vector["sarcasm-label"],np.array(sheets_dict["Sarcasm"][i])))
        #vector["task-label"]=np.vstack((vector["task-label"],np.array(sheets_dict["Bully_Label"][i])))

       
        print(vector["doc"].shape)
        #print(sheets_dict["Bully_Label"][i])
        #print(vector["task-label"].shape)


#vector["task-label"]=np.array(label)
#vector["emoti-label"]=np.array(emoti)
#vector["senti-label"]=np.array(senti)
vector["emotion-label"]=np.array(emoti)
vector["sentiment-label"]=np.array(senti)
vector["severiety-label"]=np.array(severity)
vector["complain-label"]=np.array(label)

#print(vector["task-label"])
print(vector["emotion-label"])
print(vector["sentiment-label"])
print(vector["severiety-label"])
print(vector["complain-label"])

print("saving data ...")
output=open("data/pkl/finaldatamixedtolearntest2.pkl","wb")
pickle.dump(vector,output)    
        
 




  
