# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 22:24:12 2021

@author: a_rathi
"""

import json
from json import JSONEncoder
import stanza
import time
import datetime
import torch
import gc
from collections import Counter

with open('C://Users//a_rathi//LocalDocuments//AI//res_parser//ALLRES.json') as json_file:
    data = json.load(json_file)

print("data  Len {} ".format(len(data)))
print("data 0 {} ".format(data[0]))
category = {}
alltext = []
for element in data:
    category[element['RESUME_TYPE']] = category.get(element['RESUME_TYPE'],0) + 1
    alltext += category[element['RESUME_TYPE']]
print("categories {} ".format(category))

#%%

#import spacy
gc.collect()
torch.cuda.empty_cache()
SIZE=64
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,ner', use_gpu=True, tokenize_batch_size=SIZE,
                      depparse_batch_size=SIZE,
                      batch_size=SIZE,
                      lemma_batch_size=SIZE,
                      pos_batch_size=SIZE,
                      ner_batch_size=SIZE
                      )
index = 0
start_time = time.monotonic()
#in_docs = [stanza.Document([], text=d['RESUME_TEXT']) for d in data]
#out_docs = nlp( in_docs )
word_ner_list = {}
class WordDesc:
    def __init__(self, text, upos, xpos, feats, ner):
        self.text  = text
        self.upos  = upos
        self.xpos  = xpos
        self.feats = feats
        self.ner   = ner

for element in data:
    index += 1
    if index == 421 or index == 422:
        continue
    element['NER'] = json.loads( str(nlp(element['RESUME_TEXT']))  )
    print('seconds: ', time.monotonic() - start_time, datetime.datetime.now() )
    print(index)
    start_time = time.monotonic()
    #print(element['NER'])
    if index%400==0:
        torch.cuda.empty_cache()
        gc.collect()

import torch
torch.cuda.is_available()
torch.cuda.empty_cache()
gc.collect()




#%%
import copy
import json
newdata = copy.deepcopy(data)
with open('C://Users//a_rathi//LocalDocuments//AI//res_parser//ALLRES_NER.json') as ner_json_file:
    for element in data:
        newdata
        index += 1
        if index == 421 or index == 422:
            continue
        s = json.dmumps( element )
        ner_json_file.write(s)

print(lst)

#%%

len(data)

word_cnt = {}
word_ner_dict = {}
upos = set()
xpos = set()
ner = set()
feats = set()
for element in data:
    if "NER" not in element:
        continue
    for sentence in  element["NER"]:
        for word in  sentence:
            word_cnt_set =  word_cnt.get(word["text"], set())
            word_cnt_set.add(element["RESUME_PATH"])
            word_cnt[word[ "text"]] = word_cnt_set
            upos.add(word["upos"])
            xpos.add(word["xpos"])
            ner.add(word["ner"])
            if word["text"]  not in word_ner_dict:
                word_ner_dict[ word["text"] ] =  { "text"  :  word["text"],
                                        "upos"  : Counter( { word["upos"] : 1} ),
                                        "xpos"  : Counter( { word["xpos"] : 1} ),
                                        "ner"   : Counter( { word["ner"]  : 1} ),
                                        "count" : 1
                                            }
                if  "feats" not in  word:
                    word_ner_dict[ word["text"] ]["feats"] = Counter()
                else:
                    word_ner_dict[ word["text"] ]["feats"] = Counter( { word["feats"] : 1 })
            else:
                word_ner_dict[ word["text"] ]["upos"][word["upos"]] += 1
                word_ner_dict[ word["text"] ]["xpos"][word["xpos"]] += 1
                word_ner_dict[ word["text"] ]["count"] += 1
                #print(" word_ner_list[ word['text'] ]  {}".format( word_ner_list[ word["text"] ] ) )
                if  "feats" in  word:
                    feat = word["feats"]
                    feat = feat.replace('=', '_')
                    flist = feat.split('|')
                    for f in flist:
                        feats.add(f)
                        word_ner_dict[ word["text"] ]["feats"][f] += 1

word_ner_dict['Email']
len(word_cnt['Email'])

itr = iter(word_ner_dict.items())
lst = [next(itr) for i in range(3)]
print(lst)


#%%
word_ner_list = []
for word, details in word_ner_dict.items():
    adict =  {"word" : word , "count" : details["count"], "doccount" : len(word_cnt[word])}
    for u in upos:
        val = details.get('upos', None)
        if val:
            adict["upos_" + u] =  val.get(u, 0)
        else:
            adict["upos_" + u] = 0
    for x in xpos:
        adict["xpos_" + x] = details.get(x, 0)
        val = details.get('xpos', None)
        if val:
            adict["xpos_" + x] =  val.get(x, 0)
        else:
            adict["xpos_" + x] = 0
    for n in ner:
        adict["ner_" + n] = details.get(n, 0)
        val = details.get('ner', None)
        if val:
            adict["ner_" + n] =  val.get(n, 0)
        else:
            adict["ner_" + n] =  0
    for f in feats:
        adict["feats_" + f] = details.get(f, 0)
        val = details.get('feats', None)
        if val:
            adict["feats_" + f] =  val.get(f, 0)
        else:
            adict["feats_" + f] =  0

    word_ner_list.append(adict)

#%%
word_ner_list[:10]

import pandas as pd
dataframe = pd.DataFrame.from_dict(word_ner_list)
pd.pandas.set_option('display.max_columns', None)

dataframe[ dataframe.word=='Email']
dataframe[ dataframe.word=='Email']['doccount']
word_ner_dict['Email']

df_file_name = 'C://Users//a_rathi//LocalDocuments//AI//res_parser//ALL_TOKENS_DF.feather'
dataframe.to_feather(df_file_name)

sql_file_name = 'C://Users//a_rathi//LocalDocuments//AI//res_parser//ALL_TOKENS_DF.db'
import sqlite3 as db
con = db.connect(sql_file_name)
table_name = "tokens"
dataframe.to_sql(table_name, con, if_exists="replace")
cur = con.cursor()
cur.execute("SELECT  word, \
 count, \
 doccount,\
 upos_PUNCT,\
 upos_PART,\
 upos_ADV,\
 upos_X,\
 upos_PRON,\
 upos_AUX,\
 upos_INTJ,\
 upos_CCONJ,\
 upos_NOUN,\
 upos_NUM,\
 upos_VERB,\
 upos_ADJ,\
 upos_SYM,\
 upos_SCONJ,\
 upos_ADP,\
 upos_PROPN,\
 upos_DET FROM tokens ORDER BY count desc limit 500")
names = list(map(lambda x: x[0], cur.description))
cur.fetchall()

csv_file_name = 'C://Users//a_rathi//LocalDocuments//AI//res_parser//ALL_TOKENS_DF.csv'
dataframe.to_csv(csv_file_name, header=True )

newdf = pd.read_csv(csv_file_name)

#%%
index=0
newindex=1
text_file_name = f'C://Users//a_rathi//LocalDocuments//AI//res_parser//ALL_TOKENS_{newindex}.json'
token_file = open(text_file_name, "w")
for element in data:
    if "NER" not in element:
        continue
    index+=1
    if index%500==0:
        newindex+=1
        text_file_name = f'C://Users//a_rathi//LocalDocuments//AI//res_parser//ALL_TOKENS_{newindex}.json'
        token_file.close()
        token_file = open(text_file_name, "w")
    token_file.write(json.dumps(element["NER"]) + "\n")
token_file.close()

#print(*[f'word: {word.text}\tstrat: {word.start_char}\tend: {word.end_char}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}' for sent in doc.sentences for word in sent.words], sep='\n')


#%%
import  re
phone_comp = re.compile("(\+\s?\d{1,3}\s?)?((\(\s?\d{3}\s?\)\s?)|(\d{3})(\s|-?))(\d{3}(\s|-?))(\d{4})(\s?(([E|e]xt[:|.|]?)|x|X)(\s?\d+))?" )
email_comp = re.compile("""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])""")

text = """
my num 0402191299  is this
mob:+61402191299  is this
The regex should accept numbers like

+1 8087339090
+91 8087339090
+912 8087339090
8087339090
08087339090
+1-8087339090
+91-8087339090
+912-8087339090
+918087677876(Country code(2 digits) + 10 digits Mobile Number)
+9108087735454(Country code(3 digits) + 10 digits Mobile Number)
The regex should not accept numbers like

++51 874645(double successive +)
+71 84364356(double successive spaces)
+91 808 75 74 678(not more than one space)
+91 808-75-74-678(not more than one -)
+91-846363
80873(number less than 10 digit)
8087339090456(number greater than 10 digit)
0000000000(all zeros)
+91 0000000(all zeros with country code)

"""
tl = phone_comp.findall(text)
print("".join(tl[1]))
print(tl)


sql_file_name = 'C://Users//a_rathi//LocalDocuments//AI//res_parser//ALL_TOKENS_DF.db'
import sqlite3 as db
conn = db.connect(sql_file_name)
table_name = "tokens"
cur = conn.cursor()
cur.execute("SELECT word FROM tokens")
rows = cur.fetchall()
for row in rows:
    phones = phone_comp.findall(row[0])
    if phones:
       print( ",".join([ "".join(x) for x in phones ] ) )


sql_file_name = 'C://Users//a_rathi//LocalDocuments//AI//res_parser//ALL_TOKENS_DF.db'
import sqlite3 as db
conn = db.connect(sql_file_name)
table_name = "tokens"
cur = conn.cursor()
cur.execute("SELECT word FROM tokens")
rows = cur.fetchall()
for row in rows:
    phones = email_comp.findall(row[0])
    if phones:
       print( ",".join([ "".join(x) for x in phones ] ) )



for row in data:
    for c in list(row["RESUME_TEXT"])
    if phones:
        for p in phones:
            print("match " +  "".join(p) + " : " + )




#%%


import re
def urlify(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespace with a single dash
    s = re.sub(r"\s+", '_', s)

    return s
len(data)
PATH = "C:/Users/a_rathi/LocalDocuments/AI/res_parser/Train/"
LABELS2IDX  = {
            "LINE_START" : 0,
            "LINE_MIDDLE" : 1,
            "LINE_END" : 2,

            "WORD_START" : 0,
            "WORD_MIDDLE" : 1,
            "WORD_END" : 2,
            "WORD_UNK" : 3,

            "PARA_START" : 0,
            "PARA_MIDDLE" : 1,
            "PARA_END" : 2,
            "PARA_UNK" : 2,
        }


CHAR2IDX = set()
for row in data:
    clist = list(row["RESUME_TEXT"])
     for c in clist:
           CHAR2IDX.add(c)

#==============================================================================================

def mark_last(iterable):
    try:
        *init, last = iterable
    except ValueError:  # if iterable is empty
        return
    for e in init:
        yield e, True
    yield last, False

chars_set = set()
file_list = {}
for row in data:
    fname = PATH + urlify(row["RESUME_NAME"]) + ".csv"
    clist = list(row["RESUME_TEXT"])
    para_list = []
    line_list = []
    line_text = ""
    for c in clist:
        chars_set.add(c)
        if c == '\n':
            line_text + = c
            line_list.append(line_text)
            line_text = ""
        else:
            line_text + = c
    empty_line = 0
    NEW_PARA = True
    para = []
    for l in line_list:
            NEW_PARA = False
        if len(l.strip())==0:
            empty_line+=1
            NEW_PARA = True
            para.append(l)
        else:
            if NEW_PARA:
                para_list.append(para)
                para = []
            para.append(l)
    file_list[fname] = { "PARAS" : para_list, "COUNT" : len(clist) }


char2idx =  { idx:ch  for idx, ch in enumerate(chars_set) }
for fname, props in file_list.items():
    paras  = props["PARAS"]
    atoken = ""
    with open(fname) as train_file:
        clen = props["COUNT"] + 1
        currpos = 0
        for para, doc_flag in mark_last( paras ):
            PARA = 0
            for line, para_flag in mark_last( para ):
                if para_flag == False:
                    PARA = 3
                LINE = 0
                WORD = 0
                prev_c = ' '
                for c, flag in mark_last( list(line) ):
                    posperc = round( currpos / (1.0 * clen) , 2 )
                    if WORD == 2:
                        if c.isspace() or flag==False:
                            WORD = 3
                    if c.isalnum() and prev_c.isspace():
                        WORD = 2
                    if LINE == 0:
                        train_file.write(f"{c},{PARA},{LINE},{WORD},{posperc}")
                        if c == '\n' or line_flag == False:
                            LINE = 3
                            train_file.write(f"{c},{PARA},{LINE},{WORD},{posperc}")
                        else:
                            LINE = 2
                    if LINE == 2:
                        train_file.write(f"{c},{PARA},{LINE},{WORD},{posperc}")

                    if WORD == 0:
                        WORD = 3
                if PARA == 0:
                    PARA = 1
