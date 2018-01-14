#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 21:04:20 2017

@author: satishreddychirra
"""


# INDEXING

import re
import json
import email
from glob import glob 
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from elasticsearch import Elasticsearch


# set of stopwords
stop = set(stopwords.words('english'))


# Creating a label dictionary
label_dict = {}
file = open('/Users/satishreddychirra/Downloads/trec07p/full/index1', 'r')
for line in file.readlines():
    l = line.split(' ')
    label_dict[int(str(l[1]).replace('\n',''))] = l[0]

# document split train vs test
from random import shuffle
sp = []
for i in range(75420):
    if(i <= 60336):
        sp.append('train')
    else:
        sp.append('test')
shuffle(sp)
 
   
words_u = []
es = Elasticsearch(timeout=30)
for filename in glob('/Users/satishreddychirra/Downloads/trec07p/data/*'):
    with open(filename, 'r', encoding='ISO-8859-1') as file:
        data = file.read()
    
    e_msg = email.message_from_string(data)
    
    # Extracting body of the email
    if e_msg.is_multipart():
        for part in e_msg.walk():
            content_type = part.get_content_type().lower()
            content_dis = str(part.get('Content-Disposition'))
            
            if((content_type == 'text/plain' or content_type == 'text/html') and 'attachment' not in content_dis):
                try:
                    body = str(part.get_payload(decode=True)).replace("b'","").replace("'","") # decode
                except AssertionError: 
                    print('Assertion Error for key: ' + str(int(filename.split('.')[1])))
                break
    else:
        body = str(e_msg.get_payload(decode=True)).replace("b'","").replace("'","")
           
    # Tokenizing
    clean = re.compile('<.*?>')
    text_w = re.sub(clean, '', str(body).lower())
    text_w = re.findall(r'[0-9a-z]+(?:\.?[0-9a-z]+)+', text_w)

    # Filtering the invalid words
    text_p = ''
    for i in text_w:
        if(i not in stop):
            stemmer = PorterStemmer()
            text_p += stemmer.stem(i) + " "

    # Storing the words in a list for unigrams
    for word in text_p.split(' '):
        words_u.append(word)

    key = int(filename.split('.')[1])
    
    # Loading data to elasticsearch
    docu = {
            'docno': int(str(key).encode('utf-8', '').decode('utf-8')),
            'text': text_p.strip().encode('utf-8', '').decode('utf-8'),
            'length': int(str(len(text_p.split())).encode('utf-8', '').decode('utf-8')),
            'label': str(label_dict[key]).strip().encode('utf-8', '').decode('utf-8'),
            'split': str(sp[key]).strip().encode('utf-8', '').decode('utf-8')
    }
    resu = es.index(index="sh_dataset", doc_type='document', id=key, body=json.loads(json.dumps(docu)), ignore=[400, 404])
    print(resu['created'])


words_froz = list(set(words_u))
words_froz = list(filter(None, words_froz))

count = 1
words_voc = {}
for element in words_froz:
    words_voc[element] = count
    count += 1
    

#------------------------------------------------------------------------------

manual_list = []
file = open('/Users/satishreddychirra/Document/manual_list.txt','r')
for line in file.readlines():
    manual_list.append(line.replace('\n',''))


from elasticsearch import Elasticsearch
import elasticsearch.helpers
es = Elasticsearch(timeout = 30)

feat_dict = {}
for i in range(1,75420):
    feat_dict[i] = []

for word in manual_list:     
    stemmer = PorterStemmer()
    word_s = ("%s" % (stemmer.stem(word)))
       
    res = elasticsearch.helpers.scan(es, {"_source": True, "query": {"match": {"text": word}}, "script_fields": {"index_df": {"script": {"lang": "groovy",
                                    "inline": "_index['text']['" + word_s + "'].df()"}}, "index_tf": {"script": {"lang": "groovy", "inline": "_index['text']['" + word_s + "'].tf()"}}}},
                                    index="sh_dataset", doc_type="document", scroll=u"5m")
    
    res_list = []
    for i in res:
        res_list.append(i)


    tf = []
    doc_no = []
    for i in range(len(res_list)):
        tf.append(int(res_list[i]['fields']['index_tf'][0]))
        doc_no.append(int(res_list[i]['_id']))
               
    
    temp = dict(zip(doc_no,tf))
    
    for k in feat_dict:
        if(k in temp.keys()):
            feat_dict[k].append(temp[k])
        elif(k not in temp.keys()):
            feat_dict[k].append(0)
    
    

for k in feat_dict:
    feat_dict[k].append(sp[k])

for k in feat_dict:
    if(label_dict[k] == 'spam'):
        feat_dict[k].append(0)
    elif(label_dict[k] == 'ham'):
        feat_dict[k].append(1)
     

import pandas as pd
feature_df = pd.DataFrame.from_dict(feat_dict, orient='index')


f_train, f_test = feature_df[feature_df[96] == 'train'], feature_df[feature_df[96] == 'test']

x_train,y_train = f_train[f_train.columns[0:96]], f_train[f_train.columns[97]]
x_test,y_test = f_test[f_test.columns[0:96]], f_test[f_test.columns[97]]

# fit a model
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()

model = lm.fit(x_train, y_train)
pred_test = lm.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred_test)

#----------
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

model = dt.fit(x_train, y_train)
pred_test = dt.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred_test)



for name, importance in zip(x_train.columns, model.feature_importances_):
    print(name, importance)

#------------------------------------------------------------------------------

# Part-B

# Loading the spam words
spam_list = []
file = open('/Users/satishreddychirra/Downloads/spam_list.txt','r')
for line in file.readlines():
    spam_list.append(line.replace('\n',''))


feat_dict_sl = {}
for i in range(1,75420):
    feat_dict_sl[i] = []


for word in spam_list: 
    stemmer = PorterStemmer()
    word_s = ("%s" % (stemmer.stem(word)))           
    res = elasticsearch.helpers.scan(es, {"_source": True, "query": {"match": {"text": word}}, "script_fields": {"index_df": {"script": {"lang": "groovy",
                                    "inline": "_index['text']['" + word_s + "'].df()"}}, "index_tf": {"script": {"lang": "groovy", "inline": "_index['text']['" + word_s + "'].tf()"}}}},
                                    index="sh_dataset", doc_type="document", scroll=u"5m")
    
    res_list = []
    for i in res:
        res_list.append(i)

    tf = []
    doc_no = []
    for i in range(len(res_list)):
        tf.append(int(res_list[i]['fields']['index_tf'][0]))
        doc_no.append(int(res_list[i]['_id']))               
    
    temp = dict(zip(doc_no,tf))
    
    for k in feat_dict_sl:
        if(k in temp.keys()):
            feat_dict_sl[k].append(temp[k])
        elif(k not in temp.keys()):
            feat_dict_sl[k].append(0)
    
    

for k in feat_dict_sl:
    feat_dict_sl[k].append(sp[k])

for k in feat_dict_sl:
    if(label_dict[k] == 'spam'):
        feat_dict_sl[k].append(0)
    elif(label_dict[k] == 'ham'):
        feat_dict_sl[k].append(1)


import pandas as pd
feat_sl_df = pd.DataFrame.from_dict(feat_dict_sl, orient='index')


f_train, f_test = feat_sl_df[feat_sl_df[157] == 'train'], feat_sl_df[feat_sl_df[157] == 'test']

x_train,y_train = f_train[f_train.columns[0:157]], f_train[f_train.columns[158]]
x_test,y_test = f_test[f_test.columns[0:157]], f_test[f_test.columns[158]]

# fit a model
#------
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()

model = lm.fit(x_train, y_train)
pred_test = lm.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred_test)

#------
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

model = dt.fit(x_train, y_train)
pred_test = dt.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred_test)

#------
# Top 50 spam percentage
pred_test_p = dt.predict_proba(x_test)

x_test['prediction'] = pred_test_p[:,0]
x_test = x_test.sort_values('prediction', ascending=True)

temp = list(x_test['prediction'].head(50))

count_zero = 0
count_one = 0
for i in temp:
    if(i < 0.5):
        count_zero +=1
    elif(i > 0.5):
        count_one += 1

spam_percent = count_zero/(count_zero+count_one)

#------------------------------------------------------------------------------


### Part-2 All unigrams as features
import elasticsearch.helpers

# Unigrams dictionary
words_list = []
for doc_num in range(1,75420):
    term_v = es.termvectors(index = "sh_dataset", doc_type = "document", id = doc_num, term_statistics=True)
    if(bool(term_v['term_vectors']) == True):
        temp = term_v['term_vectors']['text']['terms']
        for k in temp:
            words_list.append(k)

words_froz = list(frozenset(words_list))

count = 0
words_dict = {}
for element in words_froz:
    count += 1
    words_dict[element] = count
    

# Train and Test document numbers
train_doc = [i for i,x in enumerate(sp) if x == 'train']
test_doc = [i for i,x in enumerate(sp) if x == 'test']


# Dumping train data to a file
from operator import itemgetter
file = open('/Users/satishreddychirra/Document/feature_matrix_train.txt','a')
for doc_num in train_doc:
    term_v = es.termvectors(index = "sh_dataset", doc_type = "document", id = doc_num, term_statistics=True)
    if(bool(term_v['term_vectors']) == True):
        temp =  term_v['term_vectors']['text']['terms']       
        file.writelines(str(feat_dict[doc_num][16]) + ' ')
        l = {}
        for k in temp:
            l[words_dict[k]] = temp[k]['term_freq']
        
        sorted_list = [[j, v] for j, v in l.items()]
        sorted_list.sort(key=itemgetter(0))
        
        for i in range(len(sorted_list)):
            file.writelines(str(sorted_list[i][0]) + ':' + str(sorted_list[i][1]) + ' ')
        
        file.writelines('\n')
file.close()


# Dumping test data to a file
file = open('/Users/satishreddychirra/Document/feature_matrix_test.txt','a')
for doc_num in test_doc:
    term_v = es.termvectors(index = "sh_dataset", doc_type = "document", id = doc_num, term_statistics=True)
    if(bool(term_v['term_vectors']) == True):
        temp =  term_v['term_vectors']['text']['terms']       
        file.writelines(str(feat_dict[doc_num][16]) + ' ')
        l = {}
        for k in temp:
            l[words_dict[k]] = temp[k]['term_freq']
        
        sorted_list = [[j, v] for j, v in l.items()]
        sorted_list.sort(key=itemgetter(0))
        
        for i in range(len(sorted_list)):
            file.writelines(str(sorted_list[i][0]) + ':' + str(sorted_list[i][1]) + ' ')
            
        file.writelines('\n')
file.close()


# java -cp liblinear-2.11.jar de.bwaldvogel.liblinear.Train -s 0 -c 0.001 feature_matrix_train.txt output.txt

# java -cp liblinear-2.11.jar de.bwaldvogel.liblinear.Predict -b 1 feature_matrix_test.txt output.txt out.txt

'''
-s type : set type of solver (default 1)
  for multi-class classification
   0 -- L2-regularized logistic regression (primal)
   1 -- L2-regularized L2-loss support vector classification (dual)
   2 -- L2-regularized L2-loss support vector classification (primal)
   3 -- L2-regularized L1-loss support vector classification (dual)
   4 -- support vector classification by Crammer and Singer
   5 -- L1-regularized L2-loss support vector classification
   6 -- L1-regularized logistic regression
   7 -- L2-regularized logistic regression (dual)
'''

#------------------------------------------------------------------------------

# Top 50 predictions
pre = []
file = open('/Users/satishreddychirra/Document/out.txt','r')
for line in file.readlines():
    l = line.split(' ')
    pre.append(float(l[1]))
    
pre.sort()


count_zero = 0
count_one = 0
for i in pre[0:50]:
    if(i < 0.5):
        count_zero +=1
    elif(i > 0.5):
        count_one += 1

spam_percent = count_zero/(count_zero+count_one)





































