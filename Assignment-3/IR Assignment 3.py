
import re
import json
from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.parse import urlparse, urljoin, urlsplit, urlunparse
from nltk.corpus import stopwords
from reppy.robots import Robots
from reppy.ttl import HeaderWithDefaultPolicy
from elasticsearch import Elasticsearch

seed_url = ['http://en.wikipedia.org/wiki/List_of_modern_artists',
'http://www.ranker.com/list/famous-modernism-artists/reference',
'http://en.wikipedia.org/wiki/Pablo_Picasso',
'http://en.wikipedia.org/wiki/Category:Paintings_by_Pablo_Picasso']

from socket import timeout
from urllib.error import HTTPError, URLError
stop = set(stopwords.words('english'))

def web_crawler(url, depth, fc):

    # HTTP Headers
    headers = dict(urlopen(url, timeout = 10).info())        
    
    # HTML
    html = urlopen(url, timeout = 10).read().decode('utf-8')
    urlopen(url, timeout = 10).close()
    
    soup = BeautifulSoup(html, 'html.parser')
    # clean_html = soup.prettify()
    
    # Page Title
    t = soup.title.contents[0]
    
    # Document Number
    docno = url
    
    # Depth
    d = depth
    

    # URL Canonicalization
    out_links = [str(link.get('href')) for link in soup.findAll('a')]
    out_links = [link for link in out_links if not link.endswith(('.jpg','.pdf','.mp3', '.exe', '.ogg','.png', '.jpeg', '.svg', '.tar', '.rar', '.aspx', '.php', '.zip'))]
    out_links = [urlsplit(link).geturl() for link in out_links] # Converting scheme into lower case
    out_links = [urljoin(link, urlparse(link).netloc.lower()) for link in out_links] # Converting netloc into lowercase
    out_links = [re.sub(r'(?::\d+)?','', link) for link in out_links] # Removing the port number
    out_links = [link.split('#')[0] for link in out_links] # To Remove the fragment, which begins with #
    out_links = [urljoin(link, urlparse(link).path.replace('//', '/')) for link in out_links] # Remove duplicate slashes
    out_links = [link.replace("https://","http://") for link in out_links] # Replacing https with http
    out_links = list(set(out_links)) # Removing Duplicate URL's
    out_links = list(filter(None, out_links)) # Removing empty elements in the list
    out_links = [urljoin(url, link) for link in out_links]
    
    # Text extraction and cleaning
    text_w = ''
    for para in soup.findAll('p'):
        text_w += para.text
    text_w = re.findall(r'[0-9a-z]+(?:\.?[0-9a-z]+)+',text_w.lower())
    text_p = ''
    for i in text_w:
        if(i not in stop):
            text_p += i + ' '
    text_p = re.sub(r'\[.*?\]','',text_p)  
    
    # Write data into a file
    file = open('/Users/satishreddychirra/Document/Crawled_files/crawled_file_' + str(fc) + '.txt', 'a')
    file.write(('<DOC>' + '\n' + '<DOCNO>' + str(url) + '</DOCNO>' + '\n' + '<HTTPheader>' + '\n' + str(headers).strip() + '\n' + '</HTTPheader>' + '\n' + '<title>' + str(t) + '</title>' + '\n' + '<text>' + '\n' + str(text_p) + '\n' + '</text>' + '\n' + '<in_links>\n</in_links>\n' + 
               '<out_links>' + '\n' + ("\n".join(out_links)) + '\n</out_links>' + '\n' + '<author> Satish </author>' + '\n' + '<depth>' + str(d) + '</depth>' + '\n' + '<url>' + str(url) + '</url>' + '\n' + '<html_Source>' + '\n' + str(html)+ '\n' + '</html_Source>\n</DOC>\n').encode('ascii', 'ignore').decode('ascii'))
    file.close()
    
    # Loading data into elasticsearch
    es = Elasticsearch(timeout=30)
    docu = {
        'docno': str(url).strip().encode('utf-8', '').decode('utf-8'),
        'http_headers':str(headers).strip().encode('utf-8', '').decode('utf-8'),
        'title': str(t).encode('utf-8', '').decode('utf-8'),
        'text': str(text_p).strip().encode('utf-8', '').decode('utf-8'),
        'in_links': '',
        'out_links': ("\n".join(out_links)),
        'author': 'Satish',
        'depth': d,
        'url': str(url).strip().encode('utf-8', '').decode('utf-8'),
        'html_source': str(html).encode('utf-8', '').decode('utf-8')                     
        }
    resu = es.index(index="sample", doc_type='document', id=str(docno).encode('utf-8', '').decode('utf-8'), body=json.loads(json.dumps(docu)), ignore=[400, 404])
    print(resu['created'])
    
    return out_links



# Implementation of Queue
from collections import deque
links_q = deque()


# Adding url's to Queue when it is empty
for x in seed_url:
    links_q.append(x)


# Importing key words from file
key_words = []
file_k = open('/Users/satishreddychirra/Document/key_words_IR_A3.txt', 'r', encoding='utf-8')
for line in file_k.readlines():
    key_words.append(line.strip())





# Crawling Loop

import time, sys
from operator import itemgetter
from urllib.robotparser import RobotFileParser

domain = {}
link_p = {}
link_s = {}
crawled_links = []
uncrawled_links = []
depth = 0
i = 0

while i < len(links_q):
    rb = RobotFileParser()
    rb.set_url(links_q[i] + '/robots.txt')
    rb.read()
    if (rb.can_fetch("*", links_q[i])):
        crawled_links.append(links_q[i])
        domain[urlparse(links_q[i]).netloc.lower()] = float(time.time()) 
        
        par = urlparse(links_q[i])
        link_d = par.scheme + '://' + par.hostname
        
        
        p = HeaderWithDefaultPolicy(default=1800, minimum=600)
        r = Robots.fetch(link_d + '/robots.txt', ttl_policy = p)
        
        d = r.agent('my-user-agent').delay
        
        if(d == None):
            if(int(domain[urlparse(links_q[i]).netloc.lower()]-float(time.time())) < 1 and len(domain) != 0):
                time.sleep(1)
                try: 
                    links = web_crawler(links_q[i], depth, i)
                except (HTTPError, URLError) as error:
                    print('HTTP/URL Error' + links_q[i]) 
                except:
                    print('Error') 
                for link in links:
                    score = 0
                    for word in (link.rsplit('/', 1)[1]).replace('_', ' ').lower().split(' '):
                        if (word in key_words):
                            score = score + 1
                    if (score >= 2):
                        link_s[link] = score
                link_p[links_q[i]] = links
                i = i + 1
                print(i)                                  
                
            else:
                try:
                    links = web_crawler(links_q[i], depth, i)
                except (HTTPError, URLError) as error:
                    print('HTTP/URL Error' + links_q[i]) 
                except:
                    print('Error') 
                    
                for link in links:
                    score = 0
                    for word in (link.rsplit('/', 1)[1]).replace('_', ' ').lower().split(' '):
                        if (word in key_words):
                            score = score + 1
                    if (score >= 2):
                        link_s[link] = score
                link_p[links_q[i]] = links 
                i = i + 1                
                               
        else: 
            print('Waiting for ' + str(d) + 'seconds')
            time.sleep(d)
            try: 
                links = web_crawler(links_q[i], depth, i)
            
            except (HTTPError, URLError) as error:
                print('HTTP/URL Error' + links_q[i]) 
            except:
                print('Error')
            
            for link in links:
                score = 0
                for word in (link.rsplit('/', 1)[1]).replace('_', ' ').lower().split(' '):
                    if (word in key_words):
                        score = score + 1
                if (score >= 2):
                    link_s[link] = score
            link_p[links_q[i]] = links            
            i = i + 1
            
        
        if (i == len(links_q)):
            # Pushing URL's to Queue
            depth = depth + 1
            
            # Poping links after completing a wave
            for x in range(len(links_q)):
                links_q.popleft()
            
            score = dict(link_s)
            sorted_list = [[j, v] for j, v in score.items()]
            sorted_list.sort(key=itemgetter(1), reverse=True)
            
            sorted_links = []
            for p, v in sorted_list:
                sorted_links.append(p)
            
            for link in sorted_links:
                try: 
                    headers = dict(urlopen(link).info())
                except (HTTPError, URLError) as error:
                    print('HTTP/URL Error')  
                
                parse = urlparse(link)
                
                l1 = urlunparse(('https',parse.hostname,parse.path.replace('//','/'),parse.params,'',''))
                l2 = urlunparse(('http',parse.hostname,parse.path.replace('//','/'),parse.params,'',''))
                
                if(str(headers['Content-Type']).split(';')[0].strip(' ') == 'text/html' and l1 not in crawled_links and l2 not in crawled_links):
                    links_q.append(link)  
            link_s = {}                

        if(i > 10):
            sys.exit()
    
    else:
        uncrawled_links.append(links_q[i])





# Indexing
es = Elasticsearch(timeout=30)

from glob import glob
for filename in glob('/Users/satishreddychirra/Document/Crawled_files/*'):
    file = open(filename, encoding='ISO-8859-1')
    soup = BeautifulSoup(file.read(), "lxml")
    for doc in soup.findAll('doc'):
        link = doc.docno.text
        if(es.exists(index="team_asr", doc_type='document', id=link) == False):
            docno = doc.docno.text
            headers = doc.httpheader.text
            t = doc.title.text
            text_p = doc.find('text').text
            out_links = doc.out_links.text   
            d = doc.depth.text
            url = doc.docno.text
            html = str(doc.html_source).replace('<html_source>','').replace('</html_source>','')

            es = Elasticsearch(timeout=30)
            docu = {
                    'docno': str(url).strip().encode('utf-8', '').decode('utf-8'),
                    'http_headers':str(headers).strip().encode('utf-8', '').decode('utf-8'),
                    'title': str(t).encode('utf-8', '').decode('utf-8'),
                    'text': str(text_p).strip().encode('utf-8', '').decode('utf-8'),
                    'in_links': '',
                    'out_links': ("\n".join(out_links)),
                    'author': 'Satish',
                    'depth': d,
                    'url': str(url).strip().encode('utf-8', '').decode('utf-8'),
                    'html_source': str(html).encode('utf-8', '').decode('utf-8')                     
                    }
            resu = es.index(index="team_asr", doc_type='document', id=str(docno).encode('utf-8', '').decode('utf-8'), body=json.loads(json.dumps(docu)), ignore=[400, 404])
            print(resu['created'])
        else:
            print('Already Exists')






# Mapping in_links
in_links = []
inlink_set = {}
for link in crawled_outlinks:
    for key in crawled_outlinks:
        if(link in crawled_outlinks[key]):
            in_links.append(key)
    inlink_set[link] = in_links
    in_links = []











