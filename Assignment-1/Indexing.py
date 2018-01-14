# INDEXING

import json
global res, doc, docno, text, text_w, soup, file, avg_doc_l, total_doc_l

total_doc_l = 0
from bs4 import BeautifulSoup
from glob import glob
for filename in glob('/Users/satishreddychirra/Downloads/AP_DATA/ap89_collection/*'):
    file = open(filename, encoding='ISO-8859-1')
    soup = BeautifulSoup(file.read(), "lxml")
    for doc in soup.findAll('doc'):
        docno = doc.docno.string
        text_w=''
        for text in doc.findAll('text'):
            text_w += text.text + " "
        total_doc_l += len(text_w.split())
        from elasticsearch import Elasticsearch
        es = Elasticsearch(timeout=30)
        docu = {
            'docno': docno.strip(),
            'text': text_w.strip(),
            'length': len(text_w.split())
        }
        resu = es.index(index="ap_dataset", doc_type='document', id=docno.strip(), body=json.loads(json.dumps(docu)), ignore=[400, 404])
        print(resu['created'])


avg_doc_l = (total_doc_l/84678)
print(avg_doc_l)

