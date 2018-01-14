# Creating Inverted files and catalogs for every 2000 documents

import os
import re
import string
from nltk.corpus import stopwords
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from glob import glob 

words_fro = []
doc_no = []
doc_len = []
doc_freq = Counter()
stop_words = []
ttf = Counter()
inv_list_a = []
count = 0
su = 0
d = 0


stop_file = open('/Users/satishreddychirra/Document/stoplist.txt', 'r')
for line in stop_file:
    stop_words.append(line.strip())

for filename in glob('/Users/satishreddychirra/Downloads/AP_DATA/ap89_collection/*'):
    file = open(filename, encoding='ISO-8859-1')
    soup = BeautifulSoup(file.read(), "html.parser")
    for doc in soup.findAll('doc'):
        docno = doc.docno.string
        text_w = ''
        for text in doc.findAll('text'):
            text_w += text.text
        
        # Tokenizing
        text_w = re.findall(r'[0-9a-z]+(?:\.?[0-9a-z]+)+',text_w.lower())
        
        stop = set(stopwords.words('english'))
        
        text_p = ''
        for i in text_w:
            if(i not in stop and i not in stop_words):
                stemmer = PorterStemmer()
                text_p += stemmer.stem(i) + " "
        
        words_a = []
        for word in text_p.split():
            words_a.append(word)
        
        doc_len.append(len(words_a))
        doc_no.append(docno.strip())
            
        words_d = frozenset(words_a)
        
        df_num = [1] * len(words_d)
        doc_df = Counter(dict(zip(words_d, df_num)))
        doc_freq.update(dict(doc_df))
        
        tf_num = [0] * len(words_d)
        doc_tf = Counter(dict(zip(words_d,tf_num)))
        ttf.update(dict(doc_tf))
        
        count = count+1
        d = d + 1
        if (count <= 2000):
            for word in words_d:
                
                words_fro.append(word)
                
                word_list = text_p.split()
                tf = word_list.count(word)
                
                dic = {word: tf}
                ttf.update(dic)
                
                pos = []
                for i, j in enumerate(text_p.split()):
                    if j == word:
                        pos.append(i+1)
                
                strn = str(d) + ',' + str(tf) + ',' + str(pos)                
                inv_list_a.append((word, strn))
                       
        if (count == 2000): 
            
            inv_list = defaultdict(list)
            for p, v in inv_list_a:
                inv_list[p].append(v)
            
            catalog = {}   
            inv_file = open('/Users/satishreddychirra/Document/inverted_file.txt', 'a')
            for p,v in inv_list.items():
                pos_b = inv_file.tell()
                con = (str(p) + ": " + str(sorted(v, key =len, reverse = True))).replace(' ','')
                inv_file.writelines(con + '\n')
                catalog.update({p : [pos_b, len(con)]})
            inv_file.close() 
            
            su = su+1
            
            cata = 'catalog_'+str(su)
            print(cata)
            vars()[cata] = catalog.copy()
            file = open('/Users/satishreddychirra/Document/inverted_file_'+ str(su) +'.txt', 'a')
            file_n = open('/Users/satishreddychirra/Document/inverted_file.txt', 'r')
            for line in file_n.readlines():
                file.writelines(line)
            file.close()
            file_n.close()
            os.remove('/Users/satishreddychirra/Document/inverted_file.txt')
            
            count = 0
            inv_list_a = []
            continue


words_froz = frozenset(words_fro)
rank = list(range(1, len(words_froz)+1))
word_id_map = dict(zip(words_froz, rank))
id_word_map = dict(zip(rank, words_froz))

docid = list(range(1, len(doc_no)+1))
doc_id = dict(zip(docid,doc_no))
doc_2id = dict(zip(doc_no,docid))

docid = list(range(1, len(doc_len)+1))
doc_info = dict(zip(docid,doc_len))

#-----------------------------------------------------------------------------------------------------------------------------------------

# Merging Files

def mergfunc(catalog_m, file_m, catalog_a, file_a, catalog_b, file_b):
    inv_file_3 = open('/Users/satishreddychirra/Document/' + file_m + '.txt', 'a')
    for k in catalog_a.keys():
        if(k in catalog_b.keys()):
            inv_file_1 = open('/Users/satishreddychirra/Document/' + file_a + '.txt', 'r')
            inv_file_1.seek(catalog_a[k][0]+len(k)+1)
            a = inv_file_1.read(catalog_a[k][1]-len(k)-1)
            inv_file_2 = open('/Users/satishreddychirra/Document/' + file_b + '.txt', 'r')
            inv_file_2.seek(catalog_b[k][0]+len(k)+1)
            b = inv_file_2.read(catalog_b[k][1]-len(k)-1)
            ab_comb = (a[1:-1] + ',' + b[1:-1]).replace("','","';'").replace("'","")
            ab_temp = []
            for element in ab_comb.split(';'):
                ab_temp.append(element)
            ab_sort = str(sorted(ab_temp, key=len, reverse=True)).replace(' ','') 
            pos_b = inv_file_3.tell()
            m_str = (k + ':' + ab_sort)
            inv_file_3.writelines(m_str + '\n')
            catalog_m.update({k : [pos_b, len(m_str)]})
        elif(k not in catalog_b.keys()):
            inv_file_1 = open('/Users/satishreddychirra/Document/' + file_a + '.txt', 'r')
            inv_file_1.seek(catalog_a[k][0]+len(k)+1)
            a = inv_file_1.read(catalog_a[k][1]-len(k)-1)
            pos_b = inv_file_3.tell()
            a_str = (k + ':' + a)
            inv_file_3.writelines(a_str + '\n')
            catalog_m.update({k : [pos_b, len(a_str)]})
    inv_file_3.close()

    inv_file_3 = open('/Users/satishreddychirra/Document/' + file_m + '.txt', 'a')
    for k in catalog_b.keys():
        if(k not in catalog_m.keys()):
            inv_file_2 = open('/Users/satishreddychirra/Document/' + file_b + '.txt', 'r')
            inv_file_2.seek(catalog_b[k][0]+len(k)+1)
            b = inv_file_2.read(catalog_b[k][1]-len(k)-1)
            pos_b = inv_file_3.tell()
            b_str = (k + ':' + b)
            inv_file_3.writelines(b_str + '\n')   
            catalog_m.update({k : [pos_b, len(b_str)]})  
    inv_file_3.close()
 


# Merging function call
# Level #1
catalog_m1 = {}
mergfunc(catalog_m1, 'inverted_file_m1', catalog_1, 'inverted_file_1', catalog_2, 'inverted_file_2')
catalog_m2 = {}
mergfunc(catalog_m2, 'inverted_file_m2', catalog_3, 'inverted_file_3', catalog_4, 'inverted_file_4')
catalog_m3 = {}
mergfunc(catalog_m3, 'inverted_file_m3', catalog_5, 'inverted_file_5', catalog_6, 'inverted_file_6')
catalog_m4 = {}
mergfunc(catalog_m4, 'inverted_file_m4', catalog_7, 'inverted_file_7', catalog_8, 'inverted_file_8')
catalog_m5 = {}
mergfunc(catalog_m5, 'inverted_file_m5', catalog_9, 'inverted_file_9', catalog_10, 'inverted_file_10')
catalog_m6 = {}
mergfunc(catalog_m6, 'inverted_file_m6', catalog_11, 'inverted_file_11', catalog_12, 'inverted_file_12')
catalog_m7 = {}
mergfunc(catalog_m7, 'inverted_file_m7', catalog_13, 'inverted_file_13', catalog_14, 'inverted_file_14')
catalog_m8 = {}
mergfunc(catalog_m8, 'inverted_file_m8', catalog_15, 'inverted_file_15', catalog_16, 'inverted_file_16')
catalog_m9 = {}
mergfunc(catalog_m9, 'inverted_file_m9', catalog_17, 'inverted_file_17', catalog_18, 'inverted_file_18')
catalog_m10 = {}
mergfunc(catalog_m10, 'inverted_file_m10', catalog_19, 'inverted_file_19', catalog_20, 'inverted_file_20')


catalog_m11 = {}
mergfunc(catalog_m11, 'inverted_file_m11', catalog_21, 'inverted_file_21', catalog_22, 'inverted_file_22')
catalog_m12 = {}
mergfunc(catalog_m12, 'inverted_file_m12', catalog_23, 'inverted_file_23', catalog_24, 'inverted_file_24')
catalog_m13 = {}
mergfunc(catalog_m13, 'inverted_file_m13', catalog_25, 'inverted_file_25', catalog_26, 'inverted_file_26')
catalog_m14 = {}
mergfunc(catalog_m14, 'inverted_file_m14', catalog_27, 'inverted_file_27', catalog_28, 'inverted_file_28')
catalog_m15 = {}
mergfunc(catalog_m15, 'inverted_file_m15', catalog_29, 'inverted_file_29', catalog_30, 'inverted_file_30')
catalog_m16 = {}
mergfunc(catalog_m16, 'inverted_file_m16', catalog_31, 'inverted_file_31', catalog_32, 'inverted_file_32')
catalog_m17 = {}
mergfunc(catalog_m17, 'inverted_file_m17', catalog_33, 'inverted_file_33', catalog_34, 'inverted_file_34')
catalog_m18 = {}
mergfunc(catalog_m18, 'inverted_file_m18', catalog_35, 'inverted_file_35', catalog_36, 'inverted_file_36')
catalog_m19 = {}
mergfunc(catalog_m19, 'inverted_file_m19', catalog_37, 'inverted_file_37', catalog_38, 'inverted_file_38')
catalog_m20 = {}
mergfunc(catalog_m20, 'inverted_file_m20', catalog_39, 'inverted_file_39', catalog_40, 'inverted_file_40')


catalog_m21 = {}
mergfunc(catalog_m21, 'inverted_file_m21', catalog_41, 'inverted_file_41', catalog_42, 'inverted_file_42')



# Level #2
catalog_g1 = {}
mergfunc(catalog_g1, 'inverted_file_g1', catalog_m1, 'inverted_file_m1', catalog_m2, 'inverted_file_m2')
catalog_g2 = {}
mergfunc(catalog_g2, 'inverted_file_g2', catalog_m3, 'inverted_file_m3', catalog_m4, 'inverted_file_m4')
catalog_g3 = {}
mergfunc(catalog_g3, 'inverted_file_g3', catalog_m5, 'inverted_file_m5', catalog_m6, 'inverted_file_m6')
catalog_g4 = {}
mergfunc(catalog_g4, 'inverted_file_g4', catalog_m7, 'inverted_file_m7', catalog_m8, 'inverted_file_m8')
catalog_g5 = {}
mergfunc(catalog_g5, 'inverted_file_g5', catalog_m9, 'inverted_file_m9', catalog_m10, 'inverted_file_m10')
catalog_g6 = {}
mergfunc(catalog_g6, 'inverted_file_g6', catalog_m11, 'inverted_file_m11', catalog_m12, 'inverted_file_m12')
catalog_g7 = {}
mergfunc(catalog_g7, 'inverted_file_g7', catalog_m13, 'inverted_file_m13', catalog_m14, 'inverted_file_m14')
catalog_g8 = {}
mergfunc(catalog_g8, 'inverted_file_g8', catalog_m15, 'inverted_file_m15', catalog_m16, 'inverted_file_m16')
catalog_g9 = {}
mergfunc(catalog_g9, 'inverted_file_g9', catalog_m17, 'inverted_file_m17', catalog_m18, 'inverted_file_m18')
catalog_g10 = {}
mergfunc(catalog_g10, 'inverted_file_g10', catalog_m19, 'inverted_file_m19', catalog_m20, 'inverted_file_m20')



catalog_g11 = {}
mergfunc(catalog_g11, 'inverted_file_g11', catalog, 'inverted_file', catalog_m21, 'inverted_file_m21')



# Level #3
catalog_r1 = {}
mergfunc(catalog_r1, 'inverted_file_r1', catalog_g1, 'inverted_file_g1', catalog_g2, 'inverted_file_g2')
catalog_r2 = {}
mergfunc(catalog_r2, 'inverted_file_r2', catalog_g3, 'inverted_file_g3', catalog_g4, 'inverted_file_g4')
catalog_r3 = {}
mergfunc(catalog_r3, 'inverted_file_r3', catalog_g5, 'inverted_file_g5', catalog_g6, 'inverted_file_g6')
catalog_r4 = {}
mergfunc(catalog_r4, 'inverted_file_r4', catalog_g7, 'inverted_file_g7', catalog_g8, 'inverted_file_g8')
catalog_r5 = {}
mergfunc(catalog_r5, 'inverted_file_r5', catalog_g9, 'inverted_file_g9', catalog_g10, 'inverted_file_g10')


# Level #4
catalog_e1 = {}
mergfunc(catalog_e1, 'inverted_file_e1', catalog_r1, 'inverted_file_r1', catalog_r2, 'inverted_file_r2')
catalog_e2 = {}
mergfunc(catalog_e2, 'inverted_file_e2', catalog_r3, 'inverted_file_r3', catalog_r4, 'inverted_file_r4')
catalog_e3 = {}
mergfunc(catalog_e3, 'inverted_file_e3', catalog_r5, 'inverted_file_r5', catalog_g11, 'inverted_file_g11')


# Level #5
catalog_b = {}
mergfunc(catalog_b, 'inverted_file_b', catalog_e1, 'inverted_file_e1', catalog_e2, 'inverted_file_e2')
catalog_f = {}
mergfunc(catalog_f, 'inverted_file_f', catalog_e3, 'inverted_file_e3', catalog_b, 'inverted_file_b')


#-----------------------------------------------------------------------------------------------------------------------------------------

# Sorting

catalog_fms = {}
inv_file_m = open('/Users/satishreddychirra/Document/inverted_file_f.txt', 'r')
for k,v in catalog_f.items():
    inv_file_m.seek(v[0]+len(k)+1)
    a_sr = inv_file_m.read(v[1]-len(k)-1)
    a_su = a_sr.replace('[','').replace(']','').replace(' ','').replace("','","';'")
    term = []
    for element in a_su.split(';'):
        y = []
        for i in element.replace("'","").split(','):
            y.append(i)
        term.append(y)
    term.sort(key=len, reverse=True)
    
    x = []
    for element in term:
        x.append(str(element).replace('[','').replace(']','').replace(' ',''))                                           
      
    inv_file_ms = open('/Users/satishreddychirra/Document/inverted_file_fms.txt', 'a')
    pos_b = inv_file_ms.tell()
    cont = (str(doc_freq[k]) + ',' + str(ttf[k]) + ":" + str(x).replace(' ','').replace('","','";"').replace('"','').replace("'","")[1:-1])
    inv_file_ms.write(cont + '\n')
    catalog_fms.update({k : [pos_b, len(cont)]})
    inv_file_ms.close() 

    
#-----------------------------------------------------------------------------------------------------------------------------------------


# OKAPI TF Calculation for all queries


results = {}
import re
from collections import Counter
from nltk.corpus import stopwords
from operator import itemgetter
from nltk.stem import PorterStemmer, WordNetLemmatizer

avg_doc_l = (sum(doc_len)/len(doc_len))
query_file = open('/Users/satishreddychirra/Document/query')
for line in query_file.readlines():
    query = line.strip()
    sen = query.lower()
    words = re.findall(r'\w+', sen, flags=re.UNICODE)
    qry = {"Query No": words[0], "Query Terms": words[1:]}
    qry_score = Counter()
    for word in qry['Query Terms']:
        
        stemmer = PorterStemmer()
        lemmatiser = WordNetLemmatizer()
        word_s = ("%s" % (stemmer.stem(word)))
        
        inv_file_m = open('/Users/satishreddychirra/Document/inverted_file_fms.txt', 'r')
        inv_file_m.seek(catalog_fms[word_s][0])
        a_sr = inv_file_m.readline()
        a_su = a_sr.split(':')
        term = [[int(y) for y in x.split(',')] for x in a_su[1].split(';')]           
        
        tf = []  
        docu_len = []
        tf_nor = []
        doc_nu = []
        for i in range(len(term)):
            tf.append(term[i][1])
            docu_len.append(doc_info[term[i][0]])
            doc_nu.append(doc_id[term[i][0]])
            tf_nor.append(tf[i] / (tf[i] + 0.5 + 1.5 * (docu_len[i]/avg_doc_l)))
            
        results.clear()
        results = dict(zip(doc_nu, tf_nor))

        qry_score = Counter(results) + qry_score
    
    score = dict(qry_score)
    score_list = [[i, v] for i, v in score.items()]
    score_list.sort(key=itemgetter(1), reverse=True)
    
    rank = list(range(1, 1001))

    results_file = open('/Users/satishreddychirra/Document/okapitf_results_a2.txt', 'a')
    for i in range(1000):
        results_file.writelines( str(qry['Query No']) + " " + str("Q0") + " " + str(score_list[i][0]) + " " + str(rank[i]) + " " + str(score_list[i][1]) + " " + str("Exp"))
        results_file.write('\n')
    results_file.close()       


#------------------------------------------------------------------------------


# BM25 Score Calculation for all queries

results = {}
D = len(doc_len)
import math
import re
from collections import Counter
from nltk.corpus import stopwords
from operator import itemgetter
from nltk.stem import PorterStemmer, WordNetLemmatizer

avg_doc_l = (sum(doc_len)/len(doc_len))
query_file = open('/Users/satishreddychirra/Document/query')
for line in query_file.readlines():
    query = line.strip()
    sen = query.lower()
    words = re.findall(r'\w+', sen, flags=re.UNICODE)
    qry = {"Query No": words[0], "Query Terms": words[1:]}
    qry_score = Counter()
    for word in qry['Query Terms']:
        
        stemmer = PorterStemmer()
        lemmatiser = WordNetLemmatizer()
        word_s = ("%s" % (stemmer.stem(word)))
        
        inv_file_m = open('/Users/satishreddychirra/Document/inverted_file_fms.txt', 'r')
        inv_file_m.seek(catalog_fms[word_s][0])
        a_sr = inv_file_m.readline()
        a_su = a_sr.split(':')
        term = [[int(y) for y in x.split(',')] for x in a_su[1].split(';')]  

        tf = []
        doc_nu = []
        docu_len = []
        tf_nor = []
        doc_frq = []
        for i in range(len(term)):
            tf.append(term[i][1])
            docu_len.append(doc_info[term[i][0]])
            doc_nu.append(doc_id[term[i][0]])
            doc_frq.append(doc_freq[word_s])
            tf_nor.append((math.log((D+0.5)/(doc_frq[i]+0.5))) * ((tf[i] + 1.2*tf[i])/(tf[i] + 1.2*((1-0.75)+ 0.75 * (docu_len[i]/avg_doc_l)))))


        results.clear()
        results = dict(zip(doc_nu, tf_nor))

        qry_score = Counter(results) + qry_score

    
    score = dict(qry_score)
    score_list = [[i, v] for i, v in score.items()]
    score_list.sort(key=itemgetter(1), reverse=True)

    rank = list(range(1, 1001))

    results_file = open('/Users/satishreddychirra/Document/bm25_results_a2.txt', 'a')
    for i in range(1000):
        results_file.writelines( str(qry['Query No']) + " " + str("Q0") + " " + str(score_list[i][0]) + " " + str(rank[i]) + " " + str(score_list[i][1]) + " " + str("Exp"))
        results_file.write('\n')
    results_file.close()


#------------------------------------------------------------------------------

# JM Smooth Calculation

def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros


import math
import re
from collections import Counter
from nltk.corpus import stopwords

def jm_dscore(single_query):
    doc_nu = []
    d_score = []
    query = str(single_query).strip()
    sen = query.lower()
    words = re.findall(r'\w+', sen, flags=re.UNICODE)
    qry = {"Query No": words[0], "Query Terms": words[1:]}
    for word in qry['Query Terms']:       
        stemmer = PorterStemmer()
        word_s = ("%s" % (stemmer.stem(word)))
        
        inv_file_m = open('/Users/satishreddychirra/Document/inverted_file_fms.txt', 'r')
        inv_file_m.seek(catalog_fms[word_s][0])
        a_sr = inv_file_m.readline()
        a_su = a_sr.split(':')
        term = [[int(y) for y in x.split(',')] for x in a_su[1].split(';')]  
        
        for i in range(len(term)):
            doc_nu.append(doc_id[term[i][0]])
        
        d_score = zerolistmaker(len(doc_nu))
            
        dict_t = dict(zip(doc_nu,d_score))
        
    return (dict_t)



from operator import itemgetter
from nltk.stem import PorterStemmer
vocSize = len(catalog_fms)
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

    total_score = Counter(results)

    for word in qry['Query Terms']:

        stemmer = PorterStemmer()
        word_s = ("%s" % (stemmer.stem(word)))
            
        inv_file_m = open('/Users/satishreddychirra/Document/inverted_file_fms.txt', 'r')
        inv_file_m.seek(catalog_fms[word_s][0])
        a_sr = inv_file_m.readline()
        a_su = a_sr.split(':')
        term = [[int(y) for y in x.split(',')] for x in a_su[1].split(';')]  
        
        cf = ttf[word_s]
        
        tf = []
        doc_nu = []
        docu_len = []
        jm_smooth = []
        for i in range(len(term)):
            tf.append(term[i][1])
            docu_len.append(doc_info[term[i][0]])
            doc_nu.append(doc_id[term[i][0]])
            jm_smooth.append(((0.8)*(tf[i]/docu_len[i]))+((0.2)*(cf/vocSize)))

        jm_score = Counter(dict(zip(doc_nu, jm_smooth)))
        jm_d = Counter(jm_dscore(qry['Query Terms']))

        for key, value in jm_d.items():
            jm_d.update({key: (value+((0.2)*(cf/vocSize)))})

        score_t = jm_d.copy()
        score_t.update(jm_score)

        for p, v in score_t.items():
            score_t.update({p: math.log(v)})

        total_score.update(score_t)

    score = dict(total_score)
    score_list = [[i, v] for i, v in score.items()]
    score_list.sort(key=itemgetter(1), reverse = True)

    rank = list(range(1, 1001))
    results_file = open('/Users/satishreddychirra/Document/jm_smooth_results_a2.txt', 'a')
    for i in range(1000):
        results_file.writelines(str(qry['Query No']) + " " + str("Q0") + " " + str(score_list[i][0]) + " " + str(rank[i]) + " " + str(score_list[i][1]) + " " + str("Exp"))
        results_file.write('\n')
    results_file.close()



#------------------------------------------------------------------------------

# Verifying df, cf

ver_file = open('/Users/satishreddychirra/Downloads/in.0.50.txt', 'r')
words_t = ver_file.read()
for word in words_t.split():
    stemmer = PorterStemmer()
    word_s = ("%s" % (stemmer.stem(word)))

    cf = ttf[word_s]
    doc_frq = doc_freq[word_s]

    inv_file_m = open('/Users/satishreddychirra/Document/output_withStem.txt', 'a')
    out = word + ' ' + str(doc_frq) + ' ' + str(cf)
    inv_file_m.writelines(out + '\n')
    inv_file_m.close()


#------------------------------------------------------------------------------

# Proximity

import math
import re
import sys
from collections import Counter
from nltk.corpus import stopwords
from operator import itemgetter
from nltk.stem import PorterStemmer

def proxi_docs(single_query):
    query = str(single_query).strip()
    sen = query.lower()
    words = re.findall(r'\w+', sen, flags=re.UNICODE)
    qry = {"Query No": words[0], "Query Terms": words[1:]}
    doc_nu = []
    for word in qry['Query Terms']:
        
        stemmer = PorterStemmer()
        word_s = ("%s" % (stemmer.stem(word)))
        
        inv_file_m = open('/Users/satishreddychirra/Document/inverted_file_fms.txt', 'r')
        inv_file_m.seek(catalog_fms[word_s][0])
        a_sr = inv_file_m.readline()
        a_su = a_sr.split(':')
        term = [[int(y) for y in x.split(',')] for x in a_su[1].split(';')]  
        
        for i in range(len(term)):
            doc_nu.append(doc_id[term[i][0]])
    
    doc_cou = Counter(doc_nu) 
    
    dp = []
    for k in doc_cou:
        if(doc_cou[k]>1):
            dp.append(k)
    
    return(dp)
    

def min_range_window(word_pos):
    rge = []
    pos = [0] * len(word_pos)
    for x in range((sum([len(sublist) for sublist in word_pos]) - 2)):
        temp = []
        for i in range(len(word_pos)):
            for j in range(1):
                temp.append(word_pos[i][pos[i]])
        rge.append(max(temp)-min(temp))
        min_index = temp.index(min(temp))
        pos[min_index] = pos[min_index] + 1
        for i in range(len(word_pos)):
            if(pos[i] > len(word_pos[i])-1):
                pos[i] = len(word_pos[i])-1
    return min(rge)


results = {}
c = 1500
query_file = open('/Users/satishreddychirra/Document/query')
for line in query_file.readlines():
    query = line.strip()
    sen = query.lower()
    words = re.findall(r'\w+', sen, flags=re.UNICODE)
    qry = {"Query No": words[0], "Query Terms": words[1:]}
    qry_score = Counter()
    list_t = []

    for word in qry['Query Terms']:
        
        stemmer = PorterStemmer()
        word_s = ("%s" % (stemmer.stem(word)))
        print(word_s)
        
        inv_file_m = open('/Users/satishreddychirra/Document/inverted_file_fms.txt', 'r')
        inv_file_m.seek(catalog_fms[word_s][0])
        a_sr = inv_file_m.readline()
        a_su = a_sr.split(':')
        term = [[int(y) for y in x.split(',')] for x in a_su[1].split(';')] 
        
        list_t.append(term)
        
       
        
    doc_r = []
    doc_pro = proxi_docs(qry['Query Terms'])
    for i in range(len(list_t)):
        doc_r.append([x[:] for x in list_t[i] if doc_id[x[0]] in doc_pro])
    
    doc_ac = []
    for i in range(len(doc_r)):
        for x in doc_r[i]:
            doc_ac.append((x[0], x[2:]))
    
    doc_list = defaultdict(list)
    for p, v in doc_ac:
        doc_list[p].append(v)
    
    doc_list_a = {}
    for k, v in doc_list.items():
        if(len(v) > 1):
            doc_list_a[k] = v
    
    pro_score = {}  
    p_score = []
    
    for k in doc_list_a:
        mrw = min_range_window(doc_list_a[k])
        p_score = (((c-mrw)*len(doc_list_a[k]))/(doc_info[k]+vocSize))
        pro_score[doc_id[k]] = p_score
    
    qry_score.update(Counter(pro_score))
    
    score = dict(qry_score)
    score_list = [[i, v] for i, v in score.items()]
    score_list.sort(key=itemgetter(1), reverse=True)

    rank = list(range(1, len(score_list)+1))
    results_file = open('/Users/satishreddychirra/Document/proximity_results.txt', 'a')
    for i in range(len(score_list)):
        results_file.writelines(str(qry['Query No']) + " " + str("Q0") + " " + str(score_list[i][0]) + " " + str(rank[i]) + " " + str(score_list[i][1]) + " " + str("Exp"))
        results_file.write('\n')
    results_file.close()
    
