#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:49:10 2017

@author: satishreddychirra
"""

# INDEXING

import json
global res, doc, docno, text, text_w, soup, file, avg_doc_l, total_doc_l

total_doc_l = 0
from bs4 import BeautifulSoup
from glob import glob
from elasticsearch import Elasticsearch
es = Elasticsearch(timeout=30)
for filename in glob('/Users/satishreddychirra/Downloads/AP_DATA/ap89_collection/*'):
    file = open(filename, encoding='ISO-8859-1')
    soup = BeautifulSoup(file.read(), "lxml")
    for doc in soup.findAll('doc'):
        docno = doc.docno.string
        text_w=''
        for text in doc.findAll('text'):
            text_w += text.text + " "
        total_doc_l += len(text_w.split())        
        docu = {
            'docno': docno.strip(),
            'text': text_w.strip(),
            'length': len(text_w.split())
        }
        resu = es.index(index="ap_dataset", doc_type='document', id=docno.strip(), body=json.loads(json.dumps(docu)), ignore=[400, 404])
        print(resu['created'])


avg_doc_l = (total_doc_l/84678)
print(avg_doc_l)


#------------------------------------------------------------------------------

# OKAPI TF Score Calculation

file_qrels = open('/Users/satishreddychirra/Document/Information Retrieval/IR A1/qrels.adhoc.51-100.AP89.txt', 'r')
qrels = {}
for line in file_qrels.readlines():
    l = line.split(' ')
    qrels[l[0],l[2]] = int(str(l[3]).replace('\n',''))


D = 84678
feat_dict = {}
feat_list = []
results_o = {}
results_t = {}
results_b = {}
qrels_m = {}
import re
import math
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
from elasticsearch import Elasticsearch
import elasticsearch.helpers
from operator import itemgetter
es = Elasticsearch(timeout = 30)
from collections import Counter, defaultdict

query_file = open('/Users/satishreddychirra/Downloads/AP_DATA/query_desc.51-100.short.txt')
for line in query_file.readlines():
    query = line.strip()
    sen = query.lower()
    words = re.findall(r'\w+', sen, flags=re.UNICODE)
    qry = {"Query No": words[0], "Query Terms": words[1:]}    
    qry_score_o = Counter()
    qry_score_t = Counter()
    qry_score_b = Counter()
    for word in qry['Query Terms']:        
        word_s = ("%s" % (stemmer.stem(word)))
        res = elasticsearch.helpers.scan(es, {"_source": True, "query": {"match": {"text": word}},
              "script_fields": {"index_df": {"script": {"lang": "groovy", "inline": "_index['text']['" + word_s + "'].df()"}},
              "index_tf": {"script": {"lang": "groovy", "inline": "_index['text']['" + word_s + "'].tf()"}}}},
              index="ap_dataset", doc_type="document", scroll=u"5m")

        res_list = []
        for i in res:
            res_list.append(i)

        tf = []
        doc_no = []
        doc_len = []
        doc_freq = []
        oka_score = []
        tfidf_score = []
        bm_score = []
        for i in range(len(res_list)):
            tf.append(res_list[i]['fields']['index_tf'][0])
            doc_no.append(res_list[i]['_id'])
            doc_len.append(res_list[i]['_source']['length'])
            doc_freq.append(res_list[i]['fields']['index_df'][0])
            oka_score.append(tf[i] / ((tf[i] + 0.5 + 1.5 * (doc_len[i] / avg_doc_l))))
            tfidf_score.append((tf[i] / (tf[i] + 0.5 + 1.5 * (doc_len[i] / avg_doc_l))) * math.log(D/doc_freq[i]))
            bm_score.append((math.log((D+0.5)/(doc_freq[i]+0.5))) * ((tf[i] + 1.2*tf[i])/(tf[i] + 1.2*((1-0.75)+ 0.75 * (doc_len[i]/avg_doc_l)))))

        results_o.clear()
        results_o = dict(zip(doc_no, oka_score))
        qry_score_o += Counter(results_o)
        
        results_t.clear()
        results_t = dict(zip(doc_no, tfidf_score))
        qry_score_t += Counter(results_t)
        
        results_b.clear()
        results_b = dict(zip(doc_no, bm_score))
        qry_score_b += Counter(results_b)

    score_o = dict(qry_score_o)
    score_list_o = [[i, v] for i, v in score_o.items()]
    score_list_o.sort(key=itemgetter(1), reverse=True)
    
    score_t = dict(qry_score_t)
    score_list_t = [[i, v] for i, v in score_t.items()]
    score_list_t.sort(key=itemgetter(1), reverse=True)
    
    score_b = dict(qry_score_b)
    score_list_b = [[i, v] for i, v in score_b.items()]
    score_list_b.sort(key=itemgetter(1), reverse=True)   
    
    feat_list_o = []
    for i in range(len(score_list_o)):
        if(((qry["Query No"], score_list_o[i][0])) in qrels.keys()):
            feat_list_o.append((qry["Query No"], score_list_o[i][0], score_list_o[i][1]))
    
    qrels_m.clear()
    for k in score_o:
        qrels_m[qry["Query No"],k] = 0
    
    for k in qrels:
        if(k[0] == qry["Query No"] and k not in qrels_m):
            feat_list_o.append((qry["Query No"], k[1], qry_score_o[min(qry_score_o, key=qry_score_o.get)]))
            
    for i in range(1000):  
        if(((qry["Query No"], score_list_o[i][0])) not in qrels.keys() and score_list_o[i][0] in score_t.keys() and score_list_o[i][0] in score_b.keys()):
            feat_list_o.append((qry["Query No"], score_list_o[i][0], score_list_o[i][1]))
            if(len(feat_list_o) > 999):
                break

    feat_dict.clear()
    for element in feat_list_o:
        feat_dict[element[0],element[1]] = element[2]
    
    feat_list_t = []
    for i in range(len(score_list_t)):
        if(((qry["Query No"], score_list_t[i][0])) in feat_dict.keys()):
            feat_list_t.append((qry["Query No"], score_list_t[i][0], score_list_t[i][1]))
    
    qrels_m.clear()
    for element in feat_list_t:
        qrels_m[element[0], element[1]] = element[2]
    
    for k in feat_dict:
        if(k not in qrels_m.keys()):
            feat_list_t.append((k[0], k[1], feat_dict[k]))
    
    feat_list_b = []
    for i in range(len(score_list_b)):
        if(((qry["Query No"], score_list_b[i][0])) in feat_dict.keys()):
            feat_list_b.append((qry["Query No"], score_list_b[i][0], score_list_b[i][1]))
    
    qrels_m.clear()
    for element in feat_list_b:
        qrels_m[element[0], element[1]] = element[2]
    
    for k in feat_dict:
        if(k not in qrels_m.keys()):
            feat_list_b.append((k[0], k[1], feat_dict[k]))
    
    for element in feat_list_o:
        feat_list.append(element)
    for element in feat_list_t:
        feat_list.append(element)
    for element in feat_list_b:
        feat_list.append(element)


feature_dict = defaultdict(list)
for p, u, v in feat_list:
    feature_dict[p, u].append(v)


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Laplace Calculation


import re
def p_laplace(term):
    resu_list = []
    dict_t = {}
    import elasticsearch.helpers
    resu = elasticsearch.helpers.scan(es, query={"_source": True, "query": {"match": {"text": str(term)}}}, index='ap_dataset', scroll=u'1m', doc_type='document')
    for i in resu:
        resu_list.append(i)

    voc_size = es.search(index="ap_dataset", body={"aggs": {"vocabSize": {"cardinality": {"field": "text"}}}})
    vocSize = voc_size['aggregations']['vocabSize']['value']

    for i in range(0, len(resu_list)):
        doc_id = resu_list[i]['_id']
        doc_l = resu_list[i]['_source']['length']
        dic = {doc_id: ((1/(doc_l + vocSize)))}
        dict_t.update(dic)
    return (dict_t)


def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros


feat_lp = []
stemmer = PorterStemmer()
voc_size = es.search(index="ap_dataset", body={"aggs": {"vocabSize": {"cardinality": {"field": "text"}}}})
vocSize = voc_size['aggregations']['vocabSize']['value']
query_file = open('/Users/satishreddychirra/Document/query')
for line in query_file.readlines():
    query = line.strip()
    sen = query.lower()
    words = re.findall(r'\w+', sen, flags=re.UNICODE)
    qry = {"Query No": words[0], "Query Terms": words[1:]}

    ts = p_laplace(qry['Query Terms'])
    document_no = []
    for key, value in ts.items():
        document_no.append(key)

    score = zerolistmaker(len(document_no))
    results = dict(zip(document_no, score))

    temp_score = Counter(results)
    total_score = Counter(results)

    for word in qry['Query Terms']:

        word_s = ("%s" % (stemmer.stem(word)))
        res = elasticsearch.helpers.scan(es, {"_source": True, "query": {"match": {"text": word}}, "script_fields": {"index_df": {"script": {"lang": "groovy",
                                        "inline": "_index['text']['" + word_s + "'].df()"}}, "index_tf": {"script": {"lang": "groovy",
                                        "inline": "_index['text']['" + word_s + "'].tf()"}}}}, index="ap_dataset", doc_type="document", scroll=u"5m")

        res_list = []
        for i in res:
            res_list.append(i)

        tf = []
        doc_no = []
        doc_len = []
        lm_laplace = []
        for i in range(len(res_list)):
            tf.append(res_list[i]['fields']['index_tf'][0])
            doc_no.append(res_list[i]['_id'])
            doc_len.append(res_list[i]['_source']['length'])
            lm_laplace.append(((tf[i] + 1 )/ (doc_len[i] + vocSize)))

        lm_score = Counter(dict(zip(doc_no, lm_laplace)))
        p_score = Counter(p_laplace(qry['Query Terms']))

        score_t = p_score.copy()
        score_t.update(lm_score)

        for p, v in score_t.items():
            score_t.update({p: math.log(v)})

        total_score.update(score_t)

    score = dict(total_score)
    score_list = [[i, v] for i, v in score.items()]
    score_list.sort(key=itemgetter(1), reverse=True)
    
    for i in range(len(score_list)):
        if(((qry["Query No"], score_list[i][0])) in feature_dict.keys()):
            feat_lp.append((qry["Query No"], score_list[i][0], score_list[i][1]))
     
    qrels_m.clear()
    for element in feat_lp:
        qrels_m[element[0], element[1]] = element[2]
    
    for k in feature_dict:
        if(qry["Query No"] == k[0] and k not in qrels_m.keys()):
            feat_lp.append((k[0], k[1], qrels_m[min(qrels_m, key=qrels_m.get)]))



for p, u, v in feat_lp:
    feature_dict[p, u].append(v)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# JM Smoothing


def jm_dscore(term):
    resu_list = []
    dict_t = {}
    import elasticsearch.helpers
    resu = elasticsearch.helpers.scan(es, query={"_source": True, "query": {"match": {"text": str(term)}}},
                                      index='ap_dataset', scroll=u'1m', doc_type='document')
    for i in resu:
        resu_list.append(i)

    for i in range(0, len(resu_list)):
        doc_id = resu_list[i]['_id']
        dic = {doc_id: (0.0)}
        dict_t.update(dic)
    return (dict_t)

def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros

feat_jm = []
stemmer = PorterStemmer()
query_file = open('/Users/satishreddychirra/Document/query')
for line in query_file.readlines():
    query = line.strip()
    sen = query.lower()
    words = re.findall(r'\w+', sen, flags=re.UNICODE)
    qry = {"Query No": words[0], "Query Terms": words[1:]}

    ts = jm_dscore(qry['Query Terms'])
    document_no = []
    for key, value in ts.items():
        document_no.append(key)

    score = zerolistmaker(len(document_no))
    results = dict(zip(document_no, score))

    temp_score = Counter(results)
    total_score = Counter(results)

    for word in qry['Query Terms']:
           
        word_s = ("%s" % (stemmer.stem(word)))
        res = elasticsearch.helpers.scan(es, {"_source": True, "query": {"match": {"text": word}}, "script_fields": {"index_ttf": {"script": {"lang": "groovy",
                                        "inline": "_index['text']['" + word_s + "'].ttf()"}}, "index_tf": {"script": {"lang": "groovy",
                                        "inline": "_index['text']['" + word_s + "'].tf()"}}}}, index="ap_dataset", doc_type="document", scroll=u"5m")

        res_list = []
        for i in res:
            res_list.append(i)

        tf = []
        doc_no = []
        doc_len = []
        cf = []
        jm_smooth = []
        for i in range(len(res_list)):
            tf.append(res_list[i]['fields']['index_tf'][0])
            doc_no.append(res_list[i]['_id'])
            doc_len.append(res_list[i]['_source']['length'])
            cf.append(res_list[i]['fields']['index_ttf'][0])
            jm_smooth.append(((0.8)*(tf[i]/doc_len[i])) + ((0.2) * (cf[i]/vocSize)))

        jm_score = Counter(dict(zip(doc_no, jm_smooth)))
        jm_d = Counter(jm_dscore(qry['Query Terms']))

        for key, value in jm_d.items():
            jm_d.update({key: (value+((0.2) * (cf[0]/vocSize)))})

        score_t = jm_d.copy()
        score_t.update(jm_score)

        for p, v in score_t.items():
            score_t.update({p: math.log(v)})

        total_score.update(score_t)

    score = dict(total_score)
    score_list = [[i, v] for i, v in score.items()]
    score_list.sort(key=itemgetter(1), reverse=True)
    
    for i in range(len(score_list)):
        if(((qry["Query No"], score_list[i][0])) in feature_dict.keys()):
            feat_jm.append((qry["Query No"], score_list[i][0], score_list[i][1]))
    
    qrels_m.clear()
    for element in feat_jm:
        qrels_m[element[0], element[1]] = element[2]
    
    for k in feature_dict:
        if(qry["Query No"] == k[0] and k not in qrels_m.keys()):
            feat_jm.append((k[0], k[1], qrels_m[min(qrels_m, key=qrels_m.get)]))


for p, u, v in feat_jm:
    feature_dict[p, u].append(v)


for key in feature_dict:
    if(key in qrels.keys()):
        feature_dict[key].append(qrels[key])
    else:
        feature_dict[key].append(0)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np

file = open('/Users/satishreddychirra/Document/feature_matrix.txt', 'a')
fd = {}
fd_list = []
for k in feature_dict:
    if(len(feature_dict[k]) == 6):
        fd[k] = tuple(feature_dict[k])
        fd_list.append((k[0], k[1], feature_dict[k][0], feature_dict[k][1], feature_dict[k][2], feature_dict[k][3], feature_dict[k][4], feature_dict[k][5]))
        file.writelines(str(k[0]) + ' ' + str(k[1]) + ' ' + str(feature_dict[k][0]) + ' ' + str(feature_dict[k][1]) + ' ' + str(feature_dict[k][2]) + ' ' + str(feature_dict[k][3]) + ' ' + str(feature_dict[k][4]) + ' ' + str(feature_dict[k][5]))
        file.writelines('\n')
file.close()  
      
fd_list = np.array(fd_list)
feature_df = pd.DataFrame(fd_list)



# Creating folds

f1_train, f1_test = feature_df[feature_df[0].isin(['85', '59', '56', '71', '64', '62', '93', '99', '58', '77', '54', '87', '94', '100', '89', '61', '95', '68', '57', '97'])], feature_df[feature_df[0].isin(['98', '60', '80', '63', '91'])]       
f2_train, f2_test = feature_df[feature_df[0].isin(['85', '59', '56', '71', '64', '62', '93', '99', '58', '77', '54', '87', '94', '100', '89', '98', '60', '80', '63', '91'])], feature_df[feature_df[0].isin(['61', '95', '68', '57', '97'])]
f3_train, f3_test = feature_df[feature_df[0].isin(['85', '59', '56', '71', '64', '62', '93', '99', '58', '77', '61', '95', '68', '57', '97', '98', '60', '80', '63', '91'])], feature_df[feature_df[0].isin(['54', '87', '94', '100', '89'])]
f4_train, f4_test = feature_df[feature_df[0].isin(['85', '59', '56', '71', '64', '54', '87', '94', '100', '89', '61', '95', '68', '57', '97', '98', '60', '80', '63', '91'])], feature_df[feature_df[0].isin(['62', '93', '99', '58', '77'])]
f5_train, f5_test = feature_df[feature_df[0].isin(['62', '93', '99', '58', '77', '54', '87', '94', '100', '89', '61', '95', '68', '57', '97', '98', '60', '80', '63', '91'])], feature_df[feature_df[0].isin(['85', '59', '56', '71', '64'])]


# Dividing into features and labels
x1_train,y1_train = f1_train[f1_train.columns[0:7]], f1_train[f1_train.columns[7]]
x1_test,y1_test = f1_test[f1_test.columns[0:7]], f1_test[f1_test.columns[7]]

x2_train,y2_train = f2_train[f2_train.columns[0:7]], f2_train[f2_train.columns[7]]
x2_test,y2_test = f2_test[f2_test.columns[0:7]], f2_test[f2_test.columns[7]]

x3_train,y3_train = f3_train[f3_train.columns[0:7]], f3_train[f3_train.columns[7]]
x3_test,y3_test = f3_test[f3_test.columns[0:7]], f3_test[f3_test.columns[7]]

x4_train,y4_train = f4_train[f4_train.columns[0:7]], f4_train[f4_train.columns[7]]
x4_test,y4_test = f4_test[f4_test.columns[0:7]], f4_test[f4_test.columns[7]]

x5_train,y5_train = f5_train[f5_train.columns[0:7]], f5_train[f5_train.columns[7]]
x5_test,y5_test = f5_test[f5_test.columns[0:7]], f5_test[f5_test.columns[7]]


#------------------------------------------------------------------------------

# fit a model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

model_f1 = lm.fit(x1_train.ix[:,2:], y1_train)
pred_test_f1= lm.predict(x1_test.ix[:,2:])
pred_train_f1 = lm.predict(x1_train.ix[:,2:])

x1_train['prediction'] = pred_train_f1
x1_test['prediction'] = pred_test_f1



results_file = open('/Users/satishreddychirra/Document/pred_test_f1.txt', 'a')
query_num = list(x1_test[0].unique())
for element in query_num:
    rank = 0
    temp = x1_test[x1_test[0] == element]
    temp.sort_values(by='prediction', ascending=False)
    for index, row in temp.iterrows():
        rank += 1
        results_file.writelines(str(row[0]) + ' ' + str('Q0') + ' ' + str(row[1]) + ' ' + str(rank) + ' ' + str(row['prediction']) + ' ' + str('Exp'))
        results_file.write('\n')           
results_file.close()


results_file = open('/Users/satishreddychirra/Document/pred_train_f1.txt', 'a')
query_num = list(x1_train[0].unique())
for element in query_num:
    rank = 0
    temp = x1_train[x1_train[0] == element]
    temp.sort_values(by='prediction', ascending=False)
    for index, row in temp.iterrows():
        rank += 1
        results_file.writelines(str(row[0]) + ' ' + str('Q0') + ' ' + str(row[1]) + ' ' + str(rank) + ' ' + str(row['prediction']) + ' ' + str('Exp'))
        results_file.write('\n')            
results_file.close()

#------------------------------------------------------------------------------


model_f2 = lm.fit(x2_train.ix[:,2:], y2_train)
pred_test_f2 = lm.predict(x2_test.ix[:,2:])
pred_train_f2 = lm.predict(x2_train.ix[:,2:])

x2_train['prediction'] = pred_train_f2
x2_test['prediction'] = pred_test_f2


results_file = open('/Users/satishreddychirra/Document/pred_test_f2.txt', 'a')
query_num = list(x2_test[0].unique())
for element in query_num:
    rank = 0
    temp = x2_test[x2_test[0] == element]
    temp.sort_values(by='prediction', ascending=False)
    for index, row in temp.iterrows():
        rank += 1
        results_file.writelines(str(row[0]) + ' ' + str('Q0') + ' ' + str(row[1]) + ' ' + str(rank) + ' ' + str(row['prediction']) + ' ' + str('Exp'))
        results_file.write('\n')          
results_file.close()


results_file = open('/Users/satishreddychirra/Document/pred_train_f2.txt', 'a')
query_num = list(x2_train[0].unique())
for element in query_num:
    rank = 0
    temp = x2_train[x2_train[0] == element]
    temp.sort_values(by='prediction', ascending=False)
    for index, row in temp.iterrows():
        rank += 1
        results_file.writelines(str(row[0]) + ' ' + str('Q0') + ' ' + str(row[1]) + ' ' + str(rank) + ' ' + str(row['prediction']) + ' ' + str('Exp'))
        results_file.write('\n')          
results_file.close()

#------------------------------------------------------------------------------


model_f3 = lm.fit(x3_train.ix[:,2:], y3_train)
pred_test_f3 = lm.predict(x3_test.ix[:,2:])
pred_train_f3 = lm.predict(x3_train.ix[:,2:])

x3_train['prediction'] = pred_train_f3
x3_test['prediction'] = pred_test_f3


results_file = open('/Users/satishreddychirra/Document/pred_test_f3.txt', 'a')
query_num = list(x3_test[0].unique())
for element in query_num:
    rank = 0
    temp = x3_test[x3_test[0] == element]
    temp.sort_values(by='prediction', ascending=False)
    for index, row in temp.iterrows():
        rank += 1
        results_file.writelines(str(row[0]) + ' ' + str('Q0') + ' ' + str(row[1]) + ' ' + str(rank) + ' ' + str(row['prediction']) + ' ' + str('Exp'))
        results_file.write('\n')         
results_file.close()


results_file = open('/Users/satishreddychirra/Document/pred_train_f3.txt', 'a')
query_num = list(x3_train[0].unique())
for element in query_num:
    rank = 0
    temp = x3_train[x3_train[0] == element]
    temp.sort_values(by='prediction', ascending=False)
    for index, row in temp.iterrows():
        rank += 1
        results_file.writelines(str(row[0]) + ' ' + str('Q0') + ' ' + str(row[1]) + ' ' + str(rank) + ' ' + str(row['prediction']) + ' ' + str('Exp'))
        results_file.write('\n')      
results_file.close()

#------------------------------------------------------------------------------


model_f4 = lm.fit(x4_train.ix[:,2:], y4_train)
pred_test_f4 = lm.predict(x4_test.ix[:,2:])
pred_train_f4 = lm.predict(x4_train.ix[:,2:])

x4_train['prediction'] = pred_train_f4
x4_test['prediction'] = pred_test_f4


results_file = open('/Users/satishreddychirra/Document/pred_test_f4.txt', 'a')
query_num = list(x4_test[0].unique())
for element in query_num:
    rank = 0
    temp = x4_test[x4_test[0] == element]
    temp.sort_values(by='prediction', ascending=False)
    for index, row in temp.iterrows():
        rank += 1
        results_file.writelines(str(row[0]) + ' ' + str('Q0') + ' ' + str(row[1]) + ' ' + str(rank) + ' ' + str(row['prediction']) + ' ' + str('Exp'))
        results_file.write('\n')         
results_file.close()


results_file = open('/Users/satishreddychirra/Document/pred_train_f4.txt', 'a')
query_num = list(x4_train[0].unique())
for element in query_num:
    rank = 0
    temp = x4_train[x4_train[0] == element]
    temp.sort_values(by='prediction', ascending=False)
    for index, row in temp.iterrows():
        rank += 1
        results_file.writelines(str(row[0]) + ' ' + str('Q0') + ' ' + str(row[1]) + ' ' + str(rank) + ' ' + str(row['prediction']) + ' ' + str('Exp'))
        results_file.write('\n')         
results_file.close()

#------------------------------------------------------------------------------

model_f5 = lm.fit(x5_train.ix[:,2:], y5_train)
pred_test_f5 = lm.predict(x5_test.ix[:,2:])
pred_train_f5 = lm.predict(x5_train.ix[:,2:])

x5_train['prediction'] = pred_train_f5
x5_test['prediction'] = pred_test_f5


results_file = open('/Users/satishreddychirra/Document/pred_test_f5.txt', 'a')
query_num = list(x5_test[0].unique())
for element in query_num:
    rank = 0
    temp = x5_test[x5_test[0] == element]
    temp.sort_values(by='prediction', ascending=False)
    for index, row in temp.iterrows():
        rank += 1
        results_file.writelines(str(row[0]) + ' ' + str('Q0') + ' ' + str(row[1]) + ' ' + str(rank) + ' ' + str(row['prediction']) + ' ' + str('Exp'))
        results_file.write('\n')        
results_file.close()


results_file = open('/Users/satishreddychirra/Document/pred_train_f5.txt', 'a')
query_num = list(x5_train[0].unique())
for element in query_num:
    rank = 0
    temp = x5_train[x5_train[0] == element]
    temp.sort_values(by='prediction', ascending=False)
    for index, row in temp.iterrows():
        rank += 1
        results_file.writelines(str(row[0]) + ' ' + str('Q0') + ' ' + str(row[1]) + ' ' + str(rank) + ' ' + str(row['prediction']) + ' ' + str('Exp'))
        results_file.write('\n')         
results_file.close()
 


































































































