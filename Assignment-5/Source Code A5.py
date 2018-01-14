#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:18:00 2017

@author: satishreddychirra
"""


# Trec Eval

file_qrels = open('/Users/satishreddychirra/Document/Information Retrieval/IR A1/qrels.adhoc.51-100.AP89.txt', 'r')

qrels = {}
for line in file_qrels.readlines():
    l = line.split(' ')
    qrels[l[0],l[2]] = int(str(l[3]).replace('\n',''))
    



# Loading the bm25 file to a list
file_bm25 = open('/Users/satishreddychirra/Document/Information Retrieval/IR A1/bm25_results.txt', 'r')

bm25 = []
for line in file_bm25.readlines():
    l = line.split(' ')
    bm25.append((l[0],l[2]))


# Finding relavent and not-relavent documents
score = []
for i in bm25:
    if i in qrels.keys():
        score.append(i + (qrels[i],))
    elif i not in qrels.keys():
        score.append(i + (0,))

# Query Numbers
file_bm25 = open('/Users/satishreddychirra/Document/Information Retrieval/IR A1/bm25_results.txt', 'r')
query_no = []
for line in file_bm25.readlines():
    l = line.split(' ')
    query_no.append(l[0])

query_no = list(set(query_no))


r_precision = 0
average_precision = 0
ndcg_t = 0
import math
for n in query_no:
    # Calculating Recall
    # Finding 'R'
    R = 0
    for i in qrels:
        if(i[0] == n):
           R += qrels[i]
    
    recall = []
    count = 0
    for i in score:
        if(i[0] == n):
            if(i[2] == 1):
                count += 1
            recall.append((1/R)*count)
                
    # Calculating Precison
    precision = []
    count = 0
    k = 0
    for i in score:
        if(i[0] == n):
            k += 1
            if(i[2] == 1):
                count += 1
            precision.append((1/k)*count)
        
    # R-Precision
    for i in range(len(precision)):
        if(precision[i] == recall[i]):
            r_pre = precision[i]
    
    r_precision += r_pre
    
    # Average Precision    
    precision_score = []
    count = 0
    k = 0
    for i in score:
        if(i[0] == n):
            k += 1
            if(i[2] == 1):
                count += 1
            precision_score.append(((1/k)*count,i[2]))
    
    pre = 0
    for i in precision_score:
        if(i[1] == 1):
            pre += i[0]
    
    avg_pre = ((1/R)*pre)
    
    average_precision += avg_pre
   
    # F1-Measure   
    f1_measure = []
    for i in range(len(precision)):
        if((precision[i]+recall[i]) == 0):
            f1_measure.append((2*precision[i]*recall[i])/(0.000000000001))
        elif((precision[i]+recall[i]) != 0):
            f1_measure.append((2*precision[i]*recall[i])/(precision[i]+recall[i]))
    
    # nDCG Calculation
    dcg = 0
    k = 0
    for i in score:
        if(i[0] == n):
            k += 1
            if(k == 1):
                dcg += i[2]
            if(k != 1):
                dcg += (i[2]/math.log(k,2))
    
    list_temp = [l for l in score if l[0] == n]          
    score_sorted = sorted(list_temp, key=lambda s: s[2], reverse = True)
    
    dcg_sorted = 0
    k = 0
    for i in score_sorted:
        if(i[0] == n):
            k += 1
            if(k == 1):
                dcg_sorted += i[2]
            if(k != 1):
                dcg_sorted += (i[2]/math.log(k,2))
        
    ndcg = dcg/dcg_sorted
    
    ndcg_t += ndcg

    file = open('/Users/satishreddychirra/Document/results_A5_t.txt','a')
    file.write('Query ID: ' + str(n) + '\n\n' + 'At 5 docs: \n' + 'Precision: ' + str(precision[4]) + '\n' + 'Recall: ' + str(recall[4]) + '\n' + 'F1: ' + str(f1_measure[4]) + '\n\n' + 
               'At 10 docs: \n' + 'Precision: ' + str(precision[9]) + '\n' + 'Recall: ' + str(recall[9]) + '\n' + 'F1: ' + str(f1_measure[9]) + '\n\n' +
               'At 20 docs: \n' + 'Precision: ' + str(precision[19]) + '\n' + 'Recall: ' + str(recall[19]) + '\n' + 'F1: ' + str(f1_measure[19]) + '\n\n' +
               'At 50 docs: \n' + 'Precision: ' + str(precision[49]) + '\n' + 'Recall: ' + str(recall[49]) + '\n' + 'F1: ' + str(f1_measure[49]) + '\n\n' +
               'At 100 docs: \n' + 'Precision: ' + str(precision[99]) + '\n' + 'Recall: ' + str(recall[99]) + '\n' + 'F1: ' + str(f1_measure[99]) + '\n\n' +
               'R-Precision: ' + str(r_pre) + '\n' + 'Average Precision: ' + str(avg_pre) + '\n' + 'nDCG: ' + str(ndcg) + '\n\n')
    file.close()


print('R-Precision: ' + str(r_precision/25))
print('Average Precision: ' + str(average_precision/25))
print('nDCG: ' + str(ndcg_t/25))

#------------------------------------------------------------------------------


# Creating a list for all the manual evaluation scores

file_s = open('/Users/satishreddychirra/Document/Satish.txt','r')
list_me = []
for line in file_s.readlines():
    l = line.split(' ')
    list_me.append((l[0],l[2],int(str(l[3]).replace('\n',''))))
    

file_r = open('/Users/satishreddychirra/Document/Ram.txt','r')
eval_r = []
for line in file_r.readlines():
    l = line.split(' ')
    eval_r.append(int(str(l[3].replace('\n',''))))

for i in range(len(list_me)):
    list_me[i] = list_me[i] + (eval_r[i],)

file_a = open('/Users/satishreddychirra/Document/Akhil.txt','r')
eval_a = []
for line in file_a.readlines():
    l = line.split(' ')
    eval_a.append(int(str(l[3]).replace('\n','')))

for i in range(len(list_me)):
    list_me[i] = list_me[i] + (eval_a[i],)



# Combining the score based on the majority

from statistics import mode
score_maj = []
for i in range(len(list_me)):
    if(list_me[i][2] != list_me[i][3] and list_me[i][2] != list_me[i][4] and list_me[i][3] != list_me[i][4]):
        score_maj.append(int((list_me[i][2]+list_me[i][3]+list_me[i][4])/3))
    else:
        score_maj.append(mode([list_me[i][2],list_me[i][3],list_me[i][4]]))
    

for i in range(len(list_me)):
    list_me[i] = list_me[i] + (score_maj[i],)


qrels_c = {}
for element in list_me:
    qrels_c[element[0], element[1]] = element[5]

#------------------------------------------------------------------------------

# BM25 
import re
import elasticsearch.helpers
from collections import Counter
from operator import itemgetter
from elasticsearch import Elasticsearch
es = Elasticsearch(timeout = 30)
D = 63757
results = {}

query_file = open('/Users/satishreddychirra/Document/query_A4.txt')
for line in query_file.readlines():
    query = line.strip()
    sen = query.lower()
    words = re.findall(r'\w+', sen, flags=re.UNICODE)
    qry = {"Query No": words[0], "Query Terms": words[1:]}
    qry_score = Counter()
    for word in qry['Query Terms']:
        print(word)
        res = elasticsearch.helpers.scan(es, {"_source": True, "query": {"match": {"text": word}},
                              "script_fields": {"index_df": {"script": {"lang": "groovy","inline": "_index['text']['" + word + "'].df()"}},
                              "index_tf": {"script": {"lang": "groovy","inline": "_index['text']['" + word + "'].tf()"}}}},
                              index="team_ars", doc_type="document", scroll=u"5m")
        
        res_list = []
        for i in res:
            res_list.append(i)

        
        tf = []
        doc_no = []
        doc_len = []
        bm25_score = []
        doc_freq = []
        for i in range(len(res_list)):
            tf.append(res_list[i]['fields']['index_tf'][0])
            doc_no.append(res_list[i]['_id'])
            doc_len.append(len(re.findall(r'\w+', res_list[i]['_source']['text'])))
            doc_freq.append(res_list[i]['fields']['index_df'][0])
            bm25_score.append((math.log((D+0.5)/(doc_freq[i]+0.5))) * ((tf[i] + 1.2*tf[i])/(tf[i] + 1.2*((1-0.75)+ 0.75 * (doc_len[i]/258)))))

        results.clear()
        results = dict(zip(doc_no, bm25_score))

        qry_score = Counter(results) + qry_score

    score = dict(qry_score)
    score_list = [[i, v] for i, v in score.items()]
    score_list.sort(key=itemgetter(1), reverse=True)

    rank = list(range(1, 1001))

    results_file = open('/Users/satishreddychirra/Document/bm25_results_A5.txt', 'a')
    for i in range(1000):
        results_file.writelines( str(qry['Query No']) + " " + str("Q0") + " " + str(score_list[i][0]) + " " + str(rank[i]) + " " + str(score_list[i][1]) + " " + str("Exp"))
        results_file.write('\n')
    results_file.close()





# Trec Eval for manual evaluated scores

# Loading the bm25 file to a list
file_bm25_A4 = open('/Users/satishreddychirra/Document/bm25_results_A5.txt', 'r')

bm25_a4 = []
for line in file_bm25_A4.readlines():
    l = line.split(' ')
    bm25_a4.append((l[0],l[2]))

 
# Finding relavent and not-relavent documents
score_c = []
for i in bm25_a4:
    if i in qrels_c.keys():
        score_c.append(i + (qrels_c[i],))
    elif i not in qrels_c.keys():
        score_c.append(i + (0,))

# Unique Query Numbers
query_num = []
for element in list_me:
    query_num.append(element[0])

query_num = list(set(query_num))


import math
import matplotlib.pyplot as plt
for n in query_num:
    # Calculating Recall
    # Finding 'R'
    R = 0
    for i in list_me:
        if(i[0] == n and (i[5] == 1 or i[5] == 2)):
           R += 1
    
    recall = []
    count = 0
    for i in score_c:
        if(i[0] == n):
            if(i[2] == 1 or i[2] == 2):
                count += 1
            recall.append((1/R)*count)
                
    # Calculating Precison
    precision = []
    count = 0
    k = 0
    for i in score_c:
        if(i[0] == n):
            k += 1
            if(i[2] == 1 or i[2] == 2):
                count += 1
            precision.append((1/k)*count)
        
    # R-Precision
    for i in range(len(precision)):
        if(precision[i] == recall[i]):
            r_pre = precision[i]
    
    # Average Precision    
    precision_score = []
    count = 0
    k = 0
    for i in score_c:
        if(i[0] == n):
            k += 1
            if(i[2] == 1 or i[2] == 2):
                count += 1
            precision_score.append(((1/k)*count,i[2]))
    
    pre = 0
    for i in precision_score:
        if(i[1] == 1 or i[1] == 2):
            pre += i[0]
    
    avg_pre = ((1/R)*pre)
   
    # F1-Measure   
    f1_measure = []
    for i in range(len(precision)):
        if((precision[i]+recall[i]) == 0):
            f1_measure.append((2*precision[i]*recall[i])/(0.000000000001))
        elif((precision[i]+recall[i]) != 0):
            f1_measure.append((2*precision[i]*recall[i])/(precision[i]+recall[i]))
    
    # nDCG Calculation
    dcg = 0
    k = 0
    for i in score_c:
        if(i[0] == n):
            k += 1
            if(k == 1):
                dcg += i[2]
            if(k != 1):
                dcg += (i[2]/math.log(k,2))
                
    list_temp = [l for l in score_c if l[0] == n]           
    listme_sorted = sorted(list_temp, key=lambda s: s[2], reverse = True)
    
    dcg_sorted = 0
    k = 0
    for i in listme_sorted:
        if(i[0] == n):
            k += 1
            if(k == 1):
                dcg_sorted += i[2]
            if(k != 1):
                dcg_sorted += (i[2]/math.log(k,2))
        
    ndcg = dcg/dcg_sorted
    
    # Precision-Recall plot    
    plt.plot(precision, recall)
    plt.title('Precision-Recall plot for query ' + str(n))
    plt.ylabel('Recall')
    plt.xlabel('Precision')
    plt.savefig('/Users/satishreddychirra/Document/plot_' + str(n) + '.png')
    
    # Writing the data to a file
    file = open('/Users/satishreddychirra/Document/results_eval_A5.txt','a')
    file.write('Query ID: ' + str(n) + '\n\n' + 'At 5 docs: \n' + 'Precision: ' + str(precision[4]) + '\n' + 'Recall: ' + str(recall[4]) + '\n' + 'F1: ' + str(f1_measure[4]) + '\n\n' + 
               'At 10 docs: \n' + 'Precision: ' + str(precision[9]) + '\n' + 'Recall: ' + str(recall[9]) + '\n' + 'F1: ' + str(f1_measure[9]) + '\n\n' +
               'At 20 docs: \n' + 'Precision: ' + str(precision[19]) + '\n' + 'Recall: ' + str(recall[19]) + '\n' + 'F1: ' + str(f1_measure[19]) + '\n\n' +
               'At 50 docs: \n' + 'Precision: ' + str(precision[49]) + '\n' + 'Recall: ' + str(recall[49]) + '\n' + 'F1: ' + str(f1_measure[49]) + '\n\n' +
               'At 100 docs: \n' + 'Precision: ' + str(precision[99]) + '\n' + 'Recall: ' + str(recall[99]) + '\n' + 'F1: ' + str(f1_measure[99]) + '\n\n' +
               'R-Precision: ' + str(r_pre) + '\n' + 'Average Precision: ' + str(avg_pre) + '\n' + 'nDCG: ' + str(ndcg) + '\n\n')
    file.close()


#------------------------------------------------------------------------------




































