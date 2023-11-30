import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import  mean_absolute_error
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import pandas as pd
import traceback


import os
import pickle
from han import HAN


# import EarlyStopping
import os, sys
#sys.path.append('path_to_the_module/early-stopping-pytorch')
from pytorchtools import EarlyStopping
import sys
print('get recurssion')
print(sys.getrecursionlimit())
sys.setrecursionlimit(10000)
print(sys.getrecursionlimit())
#from torchsample.callbacks import EarlyStopping


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device='cpu'
print(device)



# output=open("data/exp2/pkl/Tamilcsv.pkl","rb")
output=open("data/pkl/finaldatamixedtolearntest2.pkl","rb")



#output=open("bert-hiattcnt-senti-emo-sever_new.pkl","rb")

mbert = pickle.load(output)


# for i in range(mbert["complain-label"].shape[0]):
    # print(mbert["complain-label"][i])



# print(mbert["complain-label"].shape)
# print(mbert["sentiment-label"].shape)

doc=mbert["doc"][:10000]
sent_per_doc=mbert["sent_per_doc"][:10000]
word_per_sent=mbert["word_per_sent"][:10000]
emoti=mbert["emotion-label"][:10000]
senti=mbert["sentiment-label"][:10000]
label = mbert["complain-label"][:10000]
severity=mbert["severiety-label"][:10000]

print("test data shape calculation")


print(emoti.shape)
print(senti.shape)
print(label.shape)
print(severity.shape)
print("test data shape calculation ....@@@@@@")

print(doc)
print(sent_per_doc)
print(word_per_sent)
print(emoti)
print(senti)
print(severity)
print("task")


senti_label={"negative":0,"neutral":1,"positive":2}
emoti_label={"joy":0,"sadness":1,"anger":2,"fear":3,"surprise":4,"disgust":5,"other":6}
sever_label={"accusation":0,"blame":1,"disapproval":2,"no explicit reproach":3,"non-complaint":4}



doc_train,doc_test,task_train,task_test=train_test_split(doc,label,test_size=0.2,stratify=label,random_state=1234)
sent_per_doc_train,sent_per_doc_test,task_train,task_test=train_test_split(sent_per_doc,label,test_size=0.2,stratify=label,random_state=1234)
word_per_sent_train,word_per_sent_test,task_train,task_test=train_test_split(word_per_sent,label,test_size=0.2,stratify=label,random_state=1234)
complainLabel_train,complain_test,task_train,task_test=train_test_split(label,label,test_size=0.2,stratify=label,random_state=1234)
emoti_train,emoti_test,task_train,task_test=train_test_split(emoti,label,test_size=0.2,stratify=label,random_state=1234)
senti_train,senti_test,task_train,task_test=train_test_split(senti,label,test_size=0.2,stratify=label,random_state=1234)
severiety_train,severiety_test,task_train,task_test=train_test_split(severity,label,test_size=0.2,stratify=label,random_state=1234)


task=task_train


doc_train,doc_val,task_train,task_val=train_test_split(doc_train,task,test_size=0.1,stratify=task,random_state=1234)
sent_per_doc_train,sent_per_doc_val,task_train,task_val=train_test_split(sent_per_doc_train,task,test_size=0.1,stratify=task,random_state=1234)
word_per_sent_train,word_per_sent_val,task_train,task_val=train_test_split(word_per_sent_train,task,test_size=0.1,stratify=task,random_state=1234)
emoti_train,emoti_val,task_train,task_val=train_test_split(emoti_train,task,test_size=0.1,stratify=task,random_state=1234)
senti_train,senti_val,task_train,task_val=train_test_split(senti_train,task,test_size=0.1,stratify=task,random_state=1234)
severiety_train,severiety_val,task_train,task_val=train_test_split(severiety_train,task,test_size=0.1,stratify=task,random_state=1234)
complainLabel_train,complainLabel_val,task_train,task_val=train_test_split(complainLabel_train,task,test_size=0.1,stratify=task,random_state=1234)



train={"doc":doc_train,"sent_per_doc":sent_per_doc_train,"word_per_sent":word_per_sent_train,"label":task_train,"emotion":emoti_train,"severiety":severiety_train,"sentiment":senti_train}
val={"doc":doc_val,"sent_per_doc":sent_per_doc_val,"word_per_sent":word_per_sent_val,"label":task_val,"emotion":emoti_val,"severiety":severiety_val,"sentiment":senti_val}
test={"doc":doc_test,"sent_per_doc":sent_per_doc_test,"word_per_sent":word_per_sent_test,"label":task_test,"emotion":emoti_test,"severiety":severiety_test,"sentiment":senti_test}




print('train size')
print(len(train))
print('test')
print(len(test))

class Tweet(torch.utils.data.Dataset):
    def __init__(self,data):
        self.data=data
        
    def __len__(self):
        return self.data["doc"].shape[0]
    def __getitem__(self,idx):
    
        if(torch.is_tensor(idx)):
            idx=idx.tolist()
            
            
            
        doc=torch.tensor(self.data["doc"][idx].astype(np.float)).long().to(device)
        sent_per_doc=torch.tensor(self.data["sent_per_doc"][idx].astype(np.float)).long().to(device)
        word_per_sent=torch.tensor(self.data["word_per_sent"][idx].astype(np.float)).long().to(device)
        label=torch.tensor(self.data["label"][idx].astype(np.long)).long().to(device)
        emotion=torch.tensor(emoti_label[self.data["emotion"][idx]]).long().to(device)
        severiety = torch.tensor(sever_label[self.data["severiety"][idx]]).long().to(device)
        sentiment=torch.tensor(senti_label[self.data["sentiment"][idx]]).long().to(device)
    
        
        sample = {
            "doc":doc,
            "sent_per_doc":sent_per_doc,
            "word_per_sent":word_per_sent,
            "label":label,
            "emotion":emotion,
            "severiety":severiety,
            "sentiment":sentiment
        }
        return sample
        
        
        
        
        
 
tweet_train = Tweet(train)
dataloader_train = DataLoader(tweet_train, batch_size=32,shuffle=False, num_workers=0)

print("train_data loaded")
print(dataloader_train)

tweet_val = Tweet(val)
dataloader_val = DataLoader(tweet_val, batch_size=32,shuffle=False, num_workers=0)
print("validation_data loaded")


tweet_test = Tweet(test)
dataloader_test = DataLoader(tweet_test, batch_size=32,shuffle=False, num_workers=0,drop_last=False)



output_size = 2
exp_name = "EMNLP_MCHarm_GLAREAll_COVTrain_POLEval_complainlabel_1" ##start saving model
# pre_trn_ckp = "EMNLP_MCHarm_GLAREAll_COVTrain" # Uncomment for using pre-trained
exp_path = "path_to_saved_files/EMNLP_ModelCkpt/"+exp_name

lr=0.0001
# criterion = nn.BCELoss() #Binary case
criterion = nn.CrossEntropyLoss()
print("criterion")
print(  criterion)

#
 
n_classes=20
vocab_size=20668
embeddings=open("embed_matrix_bert.npy","rb")
#embeddings=pickle.load(embeddings)
embeddings=np.load(embeddings)

embeddings=torch.tensor(embeddings).float().to(device)

emb_size=768
fine_tune=True
word_rnn_size=128
sentence_rnn_size=128
word_rnn_layers=1
sentence_rnn_layers=1
word_att_size=256
sentence_att_size=256
dropout= 0.5
        
 model = HAN(n_classes,vocab_size,embeddings,emb_size,fine_tune,word_rnn_size,sentence_rnn_size,word_rnn_layers,sentence_rnn_layers,word_att_size,sentence_att_size,dropout)


model.to(device)
print("Test model")
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)




def train_model(model, patience, n_epochs):
    epochs = n_epochs
#     clip = 5

    train_acc_list=[]
    val_acc_list=[]
    train_loss_list=[]
    val_loss_list=[]
    
        # initialize the experiment path
    Path(exp_path).mkdir(parents=True, exist_ok=True)
    # initialize early_stopping object
    chk_file = os.path.join(exp_path, 'checkpoint_'+exp_name+'.pt')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=chk_file)


    model.train()
    for i in range(epochs):
#         total_acc_train = 0

      
        #scheduler.step()

        total_loss_train = 0
        total_train = 0
        correct_train = 0

        for data in dataloader_train:

            doc = data['doc'].to(device)
            sent_per_doc = data['sent_per_doc'].to(device)
            word_per_sent = data['word_per_sent'].to(device)
           
            sent_per_doc=sent_per_doc.squeeze()
            

            label_train = data['label'].to(device)
            emotio_train = data['emotion'].to(device)
            sentim_train = data['sentiment'].to(device)
            severie_train = data['severiety'].to(device)
         
            model.zero_grad()
       
            emoti_out,senti_out,sever_out,output = model(doc, sent_per_doc, word_per_sent)
           loss = criterion(output, label_train)+criterion(sever_out, severie_train)+criterion(senti_out,sentim_train)+criterion(emoti_out, emotio_train)

            loss.backward(retain_graph=True)
            optimizer.step()

            with torch.no_grad():
                _, predicted_train = torch.max(output.data, 1)
                _, predicted_sever_out_train = torch.max(sever_out.data, 1)
                _, predicted_emotion_out_train = torch.max(emoti_out.data, 1)
                _, predicted_sentiment_out_train = torch.max(senti_out.data, 1)
                total_train += label_train.size(0)
                total_train += emotio_train.size(0)
                total_train += sentim_train.size(0)
                total_train += severie_train.size(0)
                #total_train += emotion_train.size(0)
                correct_train += (predicted_train == label_train).sum().item()
                correct_train += (predicted_sever_out_train == severie_train).sum().item()
                correct_train += (predicted_sentiment_out_train == sentim_train).sum().item()
                correct_train += (predicted_emotion_out_train == emotio_train).sum().item()
                
                total_loss_train += loss.item()

        
        train_acc = 100 * correct_train / total_train
        train_loss = total_loss_train/total_train
        print("model .eval out")
        model.eval()
#         total_acc_val = 0
        total_loss_val = 0
        total_val = 0
        correct_val = 0

        with torch.no_grad():
            for data in dataloader_val:                
#                 Clip features...                

                
                doc = data['doc'].to(device)
                sent_per_doc = data['sent_per_doc'].to(device)
                word_per_sent = data['word_per_sent'].to(device)

                sent_per_doc=sent_per_doc.squeeze()
                
                label_val = data['label'].to(device)
                emotion_val = data['emotion'].to(device)
                sentiment_val = data['sentiment'].to(device)
                severiety_val = data['severiety'].to(device)


                model.zero_grad()
             
                emoti_out,senti_out,sever_out,output = model(doc,sent_per_doc,word_per_sent)
                val_loss = criterion(output, label_val) + (criterion(sever_out, severiety_val)) + (criterion(senti_out, sentiment_val)) + (criterion(emoti_out, emotion_val))
               
                _, predicted_val_label = torch.max(output.data, 1)
                _, predicted_val_severity = torch.max(sever_out.data, 1)
                _, predicted_val_emotion = torch.max(emoti_out.data, 1)
                _, predicted_val_sentiment = torch.max(senti_out.data, 1)

                total_val += label_val.size(0)
                total_val += emotion_val.size(0)
                total_val += sentiment_val.size(0)
                total_val += severiety_val.size(0)

                correct_val += (predicted_val_label == label_val).sum().item()
                correct_val += (predicted_val_severity == severiety_val).sum().item()
                correct_val += (predicted_val_emotion == emotion_val).sum().item()
                correct_val += (predicted_val_sentiment == sentiment_val).sum().item()

                total_loss_val += val_loss.item()


        val_acc = 100 * correct_val / total_val
        val_loss = total_loss_val/total_val

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        
        early_stopping(val_loss, model)

        
        if early_stopping.early_stop:
            print("Early stopping")
            break
            

        
        print(f'Epoch {i+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
        model.train()
        torch.cuda.empty_cache()
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(chk_file))
    
    return  model, train_acc_list, val_acc_list, train_loss_list, val_loss_list, i
        




def test_model(model):
    model.eval()
    total_test = 0
    correct_test =0
    total_acc_test = 0
    total_loss_test = 0
    outputs = []
    sentimentoutput=[]
    emotionoutputs=[]
    severityoutputs=[]
    test_labels=[]
    with torch.no_grad():
        for data in dataloader_test:

            doc = data['doc'].to(device)
            sent_per_doc = data['sent_per_doc'].to(device)
            word_per_sent = data['word_per_sent'].to(device)


            label_test = data['label'].to(device)
            emotion_test = data['emotion'].to(device)
            severity_test = data['severiety'].to(device)
            sentiment_test = data['sentiment'].to(device)

            sent_per_doc=sent_per_doc.squeeze()
            
            print(sent_per_doc.shape)
            

            emoti_out, sent_out, sever_out, output=model(doc,sent_per_doc,word_per_sent)

            outputs += list(output.cpu().data.numpy())
            sentimentoutput += list(sent_out.cpu().data.numpy())
            emotionoutputs += list(emoti_out.cpu().data.numpy())
            severityoutputs += list(sever_out.cpu().data.numpy())
            loss = criterion(output, label_test) + (criterion(sever_out, severity_test)) + (
                        criterion(sent_out, sentiment_test)) + (criterion(emoti_out, emotion_test))

            _, predicted_test = torch.max(output.data, 1)
            _, predicted_severity = torch.max(sever_out.data, 1)
            _, predicted_emotion = torch.max(emoti_out.data, 1)
            _, predicted_sentiment = torch.max(sent_out.data, 1)
            total_test += label_test.size(0)
            total_test += severity_test.size(0)
            total_test += sentiment_test.size(0)
            total_test += emotion_test.size(0)
            correct_test += (predicted_test == label_test).sum().item()
            correct_test += (predicted_severity == severity_test).sum().item()
            correct_test += (predicted_emotion == emotion_test).sum().item()
            correct_test += (predicted_sentiment == sentiment_test).sum().item()

            total_loss_test += loss.item()
 
    acc_test = 100 * correct_test / total_test
    loss_test = total_loss_test/total_test   
    
    print(f'acc: {acc_test:.4f} loss: {loss_test:.4f}')
    return sentimentoutput,emotionoutputs,severityoutputs,outputs




n_epochs = 100
# early stopping patience; how long to wait after last time validation loss improved.
patience = 10
model, train_acc_list, val_acc_list, train_loss_list, val_loss_list, epoc_num = train_model(model, patience, n_epochs)



sentimentoutput,emotionoutputs,severityoutputs,outputs = test_model(model)


y_pred=[]
for i in outputs:
    print("np.argmax(i)")
    print(np.argmax(i))
    y_pred.append(np.argmax(i))
# # np.argmax(outputs[:])
# outputs

# # Multiclass setting
test_labels=[]

for index in range(len(test["label"])):
    test_labels.append(test["label"][index])
    # test_labels.append(senti_label[test["sentiment"][index]])
    # test_labels.append(emoti_label[test["emotion"][index]])
    # test_labels.append(emoti_label[test["severiety"][index]])
    
    
"""
for index, row in test_samples_frame.iterrows():
    lab = row['labels'][0]
    if lab=="not harmful":
        test_labels.append(0)
    elif lab=="somewhat harmful":
        test_labels.append(1)
    else:
        test_labels.append(2)
"""

# In[ ]:


def calculate_mmae(expected, predicted, classes):
    NUM_CLASSES = len(classes)
    count_dict = {}
    dist_dict = {}
    for i in range(NUM_CLASSES):
        count_dict[i] = 0
        dist_dict[i] = 0.0
    for i in range(len(expected)):
        dist_dict[expected[i]] += abs(expected[i] - predicted[i])
        count_dict[expected[i]] += 1
    overall = 0.0
    for claz in range(NUM_CLASSES): 
        class_dist =  1.0 * dist_dict[claz] / count_dict[claz] 
        overall += class_dist
    overall /= NUM_CLASSES
#     return overall[0]
    return overall


# In[ ]:

print('test_labels')

print(test_labels)
print(y_pred)
rec = np.round(recall_score(test_labels, y_pred, average="weighted"),4)
prec = np.round(precision_score(test_labels, y_pred, average="weighted"),4)
f1 = np.round(f1_score(test_labels, y_pred, average="weighted"),4)
# hl = np.round(hamming_loss(test_labels, y_pred),4)
acc = np.round(accuracy_score(test_labels, y_pred),4)
#mmae = np.round(calculate_mmae(test_labels, y_pred, [0,1]),4)
#mae = np.round(mean_absolute_error(test_labels, y_pred),4)
# print("recall_score\t: ",rec)
# print("precision_score\t: ",prec)
# print("f1_score\t: ",f1)
# print("hamming_loss\t: ",hl)
# print("accuracy_score\t: ",f1)

print(classification_report(test_labels, y_pred))


# In[ ]:


print("Acc, F1, Rec, Prec")
#print(acc, f1, rec, prec, mae, mmae)
print(acc, f1, rec, prec)



rec = np.round(recall_score(test_labels, y_pred),4)
prec = np.round(precision_score(test_labels, y_pred),4)
f1 = np.round(f1_score(test_labels, y_pred),4)
# hl = np.round(hamming_loss(test_labels, y_pred),4)
acc = np.round(accuracy_score(test_labels, y_pred),4)

print(acc,f1,rec,prec)



print('<--------for  Sentiment ------------>')

sentiment_predict=[]
for i in sentimentoutput:
    # print("np.argmax(i)")
    # print(np.argmax(i))
    sentiment_predict.append(np.argmax(i))
# # np.argmax(outputs[:])
# outputs

# # Multiclass setting




test_labels_sentiment=[]

for index in range(len(test["sentiment"])):
    # test_labels_sentiment.append(test["sentiment"][index])
    test_labels_sentiment.append(senti_label[test["sentiment"][index]])

    # torch.tensor(senti_label[self.data["sentiment"][idx]]).long().to(device)
print('<------test_labels_sentiment------>')
print(test_labels_sentiment)
print(sentiment_predict)
rec_senti = np.round(recall_score(test_labels_sentiment, sentiment_predict, average="weighted"),4)
prec_senti= np.round(precision_score(test_labels_sentiment, sentiment_predict, average="weighted"),4)
f1_senti = np.round(f1_score(test_labels_sentiment, sentiment_predict, average="weighted"),4)
# hl = np.round(hamming_loss(test_labels, y_pred),4)
acc_senti = np.round(accuracy_score(test_labels_sentiment, sentiment_predict),4)
#mmae = np.round(calculate_mmae(test_labels, y_pred, [0,1]),4)
#mae = np.round(mean_absolute_error(test_labels, y_pred),4)
# print("recall_score\t: ",rec)
# print("precision_score\t: ",prec)
# print("f1_score\t: ",f1)
# print("hamming_loss\t: ",hl)
# print("accuracy_score\t: ",f1)

print(classification_report(test_labels_sentiment, sentiment_predict))


# In[ ]:


#print("Acc, F1, Rec, Prec, MAE, MMAE")
#print(acc, f1, rec, prec, mae, mmae)
print("Acc, F1, Rec, Prec")

print(acc_senti, f1_senti, rec_senti, prec_senti)


#
# rec_senti = np.round(recall_score(test_labels_sentiment, sentiment_predict,pos_label='positive', average='micro'),4)
# prec_senti = np.round(precision_score(test_labels_sentiment, sentiment_predict,pos_label='positive', average='micro'),4)
# f1_senti = np.round(f1_score(test_labels_sentiment, sentiment_predict,pos_label='positive', average='micro'),4)
# # hl = np.round(hamming_loss(test_labels, y_pred),4)
# acc_senti = np.round(accuracy_score(test_labels_sentiment, sentiment_predict  ,pos_label='positive', average='micro'),4)
#
# print(acc_senti, f1_senti, rec_senti, prec_senti)



print('<--------for  emotion ------------>')

emotion_predict=[]
for i in emotionoutputs:

    emotion_predict.append(np.argmax(i))
# # np.argmax(outputs[:])
# outputs

# # Multiclass setting




test_labels_emotion=[]

for index in range(len(test["emotion"])):
    # test_labels_sentiment.append(test["sentiment"][index])
    test_labels_emotion.append(emoti_label[test["emotion"][index]])

    # torch.tensor(senti_label[self.data["sentiment"][idx]]).long().to(device)
print('<------test_labels_emotion------>')
print(test_labels_emotion)
print(emotion_predict)
rec_emoti = np.round(recall_score(test_labels_emotion, emotion_predict, average="weighted"),4)
prec_emoti= np.round(precision_score(test_labels_emotion, emotion_predict, average="weighted"),4)
f1_emoti = np.round(f1_score(test_labels_emotion, emotion_predict, average="weighted"),4)
# hl = np.round(hamming_loss(test_labels, y_pred),4)
acc_emoti = np.round(accuracy_score(test_labels_emotion, emotion_predict),4)
#mmae = np.round(calculate_mmae(test_labels, y_pred, [0,1]),4)
#mae = np.round(mean_absolute_error(test_labels, y_pred),4)
# print("recall_score\t: ",rec)
# print("precision_score\t: ",prec)
# print("f1_score\t: ",f1)
# print("hamming_loss\t: ",hl)
# print("accuracy_score\t: ",f1)

print(classification_report(test_labels_emotion, emotion_predict))


# In[ ]:


# print("Acc, F1, Rec, Prec, MAE, MMAE")

#print(acc, f1, rec, prec, mae, mmae)
print("Acc, F1, Rec, Prec")

print(acc_emoti, f1_emoti, rec_emoti, prec_emoti)

print('<--------for  severity ------------>')

severity_predict = []
for i in severityoutputs:
    severity_predict.append(np.argmax(i))
# # np.argmax(outputs[:])
# outputs

# # Multiclass setting


test_labels_severity = []

for index in range(len(test["severiety"])):
    # test_labels_sentiment.append(test["sentiment"][index])
    test_labels_severity.append(sever_label[test["severiety"][index]])

    # torch.tensor(senti_label[self.data["sentiment"][idx]]).long().to(device)
print('<------test_labels_severiety------>')
print(test_labels_severity)
print(severity_predict)
rec_sever = np.round(recall_score(test_labels_severity, severity_predict, average="weighted"), 4)
prec_sever = np.round(precision_score(test_labels_severity, severity_predict, average="weighted"), 4)
f1_sever = np.round(f1_score(test_labels_severity, severity_predict, average="weighted"), 4)
# hl = np.round(hamming_loss(test_labels, y_pred),4)
acc_sever = np.round(accuracy_score(test_labels_severity, severity_predict), 4)
# mmae = np.round(calculate_mmae(test_labels, y_pred, [0,1]),4)
# mae = np.round(mean_absolute_error(test_labels, y_pred),4)
# print("recall_score\t: ",rec)
# print("precision_score\t: ",prec)
# print("f1_score\t: ",f1)
# print("hamming_loss\t: ",hl)
# print("accuracy_score\t: ",f1)

print(classification_report(test_labels_severity, severity_predict))

# In[ ]:


# print("Acc, F1, Rec, Prec, MAE, MMAE")
print("Acc, F1, Rec, Prec")

# print(acc, f1, rec, prec, mae, mmae)
print(acc_sever, f1_sever, rec_sever, prec_sever)
