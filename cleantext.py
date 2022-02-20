import json
from json import JSONEncoder
import spacy
import re

import time
import datetime
import torch
from collections import Counter
import spacy
from cleantext import clean
import datefinder
import dateparser


search_dates('The first artificial Earth satellite was launched on 4 October 1957.')

dfmt = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%b/%Y', '%B/%Y', '%b/%y', '%B/%y', '%Y/%b' , '%Y/%B',  '%y/%b' , '%y/%B', ]
dfmt = ['%Y-%m','%Y/%m', '%Y %m', '%m %Y', '%m/%Y', '%m-%Y',  ]

matchesdf = datefinder.find_dates(data[1]['RESUME_TEXT'], source=True, index=True)
matchesdp = dateparser.search.search_dates(data[1]['RESUME_TEXT'],add_detected_language=True)
for match in matchesdf:
    try:
        f = int(match[1])
    except:
        try:
            f = float(match[1])
        except:
            dparsed = dateparser.parse(match[1])
            if dparsed:
                print(match[2], match[1], dparsed)
print("")
for match in matchesdp:
    print(match)

import spacy.cli
#spacy.cli.download("en_core_web_lg")
#spacy.cli.download("en_core_web_sm")
from cleantext import clean
import hashlib
from copy import deepcopy

def hash_string(input):
    byte_input = input.encode()
    hash_object = hashlib.sha256(byte_input)
    return hash_object.hexdigest()

print( hash_string(input= "123dsasdsa@dasdsa.com.lol") )

def FindEmails(text:str) :
    matchs = FindEmails.regex.finditer( text )
    list_match = []
    for m in matchs:
        e = m.group(0)
        h = "EMAIL_" + hash_string(e)+  "_EMAIL"
        s = m.span()
        d2 = { e: {"start":s[0],"end":s[1],"email":e,"hash":h } }
        FindEmails.email_dict[h] = d2
        list_match.append( (e, h) )
    for l in list_match:
        text = re.sub(l[0], l[1], text)
    return text

FindEmails.regex = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+' )
FindEmails.email_dict = {}

def FindUrls(text:str ):
    matchs = FindUrls.regex.finditer(text)
    list_match = []
    for m in matchs:
        e = m.group(0)
        h = "URL_" + hash_string(e)+  "_URL"
        s = m.span()
        d2 = { e: {"start":s[0],"end":s[1],"url":e,"hash":h } }
        FindUrls.url_dict[h] = d2
        list_match.append( (e, h) )
    for l in list_match:
        text = re.sub(l[0], l[1], text)
    return text
FindUrls.regex = re.compile(r'\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b')
FindUrls.url_dict = {}




durl_dict = {}
FindEmails("123dsasdsa@dasdsa.com.lol  1dsasdsa@dasdsa.com.lol   2dsasdsa@dasdsa.com.lol 3dsasdsa@dasdsa.com.lol" )
FindUrls.url_dict = {}

ALLRES = 'C://Users//a_rathi//LocalDocuments//AI//res_parser//ALLRES.json'
with open(ALLRES ) as json_file:
    data = json.load(json_file)

print("data  Len {} ".format(len(data)))
#print("data 0 {} ".format(data[0]))

data[0].keys()
alltext = []
I = 0
new_data = []
categories = set()
for element in data:
    I = I + 1
    print(100*I/500, end = ",")
    if I > 500:
        break
    e2 = deepcopy(element)
    rt = element['RESUME_TEXT']
    rt = FindEmails(rt)
    rt = FindUrls(rt)
    rt = clean(rt,
            fix_unicode=True,
            to_ascii=True,
            lower=True,
            normalize_whitespace=True,
            no_line_breaks=False,
            strip_lines=False,
            keep_two_line_breaks=False,
            no_urls=False,
            no_emails=False,
            no_phone_numbers=False,
            no_numbers=False,
            no_digits=False,
            no_currency_symbols=False,
            no_punct=False,
            no_emoji=False,
            replace_with_url="<URL>",
            replace_with_email="<EMAIL>",
            replace_with_phone_number="<PHONE>",
            replace_with_number="<NUMBER>",
            replace_with_digit="0",
            replace_with_currency_symbol="<CUR>",
            replace_with_punct="",
            lang="en"
            )
    categories.add(e2['RESUME_TYPE'])
    e2['RESUME_TEXT'] = rt
    new_data.append(e2)

print("categories {} ".format(categories))



nlp = spacy.load("en_core_web_lg")
doc = nlp(cdata)
i = 0
for token in doc:
    i=i+1
    if i>200:
        break
    print(token.text)
