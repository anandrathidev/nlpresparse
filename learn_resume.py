import json
from json import JSONEncoder
import spacy
import re

import time
import datetime
import torch
import collections
import spacy
from cleantext import clean
import datefinder
import dateparser.search
import spacy.cli
#spacy.cli.download("en_core_web_lg")
#spacy.cli.download("en_core_web_sm")
import hashlib
from copy import deepcopy
import string
import regex
import MyDateParser

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

FILE_PATH = 'C://Users//a_rathi//LocalDocuments//AI//res_parser//ALLRES.json'

def load_data():
    with open(FILE_PATH) as json_file:
        data = json.load(json_file)
    return data
data = load_data()


dfmt = []
dfmt += ['%d-%b-%y','%d-%b-%Y','%b-%d-%y','%b-%d-%Y','%y-%b-%d','%Y-%b-%d','%Y-%d-%b',]
dfmt += ['%d/%b/%y','%d/%b/%Y','%b/%d/%y','%b/%d/%Y','%y/%b/%d','%Y/%b/%d','%Y/%d/%b',]
dfmt += ['%d %b %y','%d %b %Y','%b %d %y','%b %d %Y','%y %b %d','%Y %b %d','%Y %d %b',]

dfmt += ['%d-%B-%y','%d-%B-%Y','%B-%d-%y','%B-%d-%Y','%y-%B-%d','%Y-%B-%d','%Y-%d-%B',]
dfmt += ['%d/%B/%y','%d/%B/%Y','%B/%d/%y','%B/%d/%Y','%y/%B/%d','%Y/%B/%d','%Y/%d/%B',]
dfmt += ['%d %B %y','%d %B %Y','%B %d %y','%B %d %Y','%y %B %d','%Y %B %d','%Y %d %B',]

dfmt += ['%d-%m-%Y','%Y-%m-%d','%m-%d-%Y','%d-%m-%y','%y-%m-%d','%m-%d-%y' ]
dfmt += ['%d/%m/%Y','%Y/%m/%d','%m/%d/%Y','%d/%m/%y','%y/%m/%d','%m/%d/%y' ]
dfmt += ['%d %m %Y','%Y %m %d','%m %d %Y','%d %m %y','%y %m %d','%m %d %y' ]

dfmt += ['%b/%Y','%b/%y','%B/%Y','%B/%y','%Y/%b','%Y/%B','%y/%b','%y/%B' ]
dfmt += ['%b-%Y','%b-%y','%B-%Y','%B-%y','%Y-%b','%Y-%B','%y-%b','%y-%B' ]
dfmt += ['%b %Y','%b %y','%B %Y','%B %y','%Y %b','%Y %B','%y %b','%y %B' ]
dfmt += ["%b'%Y","%b'%y","%B'%Y","%B'%y","%Y'%b","%Y'%B","%y'%b","%y'%B" ]
dfmt += ["%b’%Y","%b’%y","%B’%Y","%B’%y","%Y’%b","%Y’%B","%y’%b","%y’%B" ]
dfmt += ['%b,%Y','%b,%y','%B,%Y','%B,%y','%Y,%b','%Y,%B','%y,%b','%y,%B' ]

dfmt += ['%Y/%m', '%m/%Y',  ]
dfmt += ['%Y-%m', '%m-%Y',  ]
dfmt += ['%Y %m', '%m %Y', ]

#dfmt += [ '%y/%m', '%m/%y', ]
#dfmt += [ '%y-%m', '%m-%y', ]

def myparse_timestamp(datestring, formats):
    d = None
    datestring = datestring.strip()
    for f in formats:
        try:
            d = datetime.datetime.strptime(datestring, f)
            break
        except:
            continue
    return d


"""
def test():
    matchesdf = datefinder.find_dates(data[0]['RESUME_TEXT'], source=True, index=True)
    list_match = []
    matchesdf = list(matchesdf)
    for match in matchesdf:
        dparsed = myparse_timestamp(match[1], formats=dfmt)

myparse_timestamp(datestring= 'jun 12', formats=dfmt)
matchesdf = datefinder.find_dates(data[0]['RESUME_TEXT'], source=True, index=True)
list_match = []
matchesdf = list(matchesdf)
for match in matchesdf:
    dparsed = myparse_timestamp(match[1], formats=dfmt)

"""

def hash_string(input):
    byte_input = input.encode()
    hash_object = hashlib.sha256(byte_input)
    return hash_object.hexdigest()

def FindEmails(text:str) :
    matchs = FindEmails.regex.finditer( text )
    list_match = []
    for m in matchs:
        e = m.group(0)
        h = "EMAIL" + hash_string(e)+  "EMAIL"
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
        h = "URL" + hash_string(e)+  "URL"
        s = m.span()
        d2 = { e: {"start":s[0],"end":s[1],"url":e,"hash":h } }
        FindUrls.url_dict[h] = d2
        list_match.append( (e, h) )
    for l in list_match:
        text = re.sub(l[0], l[1], text)
    return text
FindUrls.regex = re.compile(r'\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b')
FindUrls.url_dict = {}

def FindDates(text:str):
    matchesdf = datefinder.find_dates(text, source=True, index=True)
    list_match = []
    for match in matchesdf:
        dparsed = myparse_timestamp(match[1], formats=dfmt)
        if dparsed:
            e = match[1]
            s = match[2]
            h = "DATE" + hash_string(e)+  "DATE"
            d2 = { e: {"start":s[0],"end":s[1],"date":e,"hash":h } }
            FindDates.dates_dict[h] = d2
            list_match.append( (e, h) )
            #print(d2)
    for l in list_match:
        text = re.sub(l[0], l[1], text)
    return text
FindDates.dates_dict = {}

def zipngram(text_list,n=2):
  print(f"zipngram {n}")
  return zip(*[text_list[i:] for i in range(n)])

class Replace:
    def __init__(self, replace_with=" "):
        self.punc_dict = dict.fromkeys(string.punctuation, replace_with)
        self.punc_dict.pop('"')
        self.punc_dict.pop('$')
        self.punc_dict.pop('%')
        self.punc_dict.pop('_')
        self.punc_trans = str.maketrans(self.punc_dict)

    def replace_punct(self, text ):
        """
        Replace punctuations from ``text`` with whitespaces (or other tokens).
        """
        return text.translate( self.punc_trans )

data[0].keys()
alltext = []
I = 0
new_data = []
categories = set()
def getBiGrams(data, n, dlen = None):
    new_data = []
    new_text = ""
    I = 0
    if dlen is None:
        dlen = len(data) +1
    filtered_sentence = []
    stop_words = set(stopwords.words('english'))
    repl = Replace()
    ngrams = {}
    for x in range(n+1):
        ngrams[x] = []

    for element in data:
        I = I + 1
        print(100*I/dlen, end = ",")
        if I > dlen:
            break
        rt = element['RESUME_TEXT']
        if len(rt)>25182:
            continue
        rt = FindEmails(rt)
        rt = FindUrls(rt)
        rt = FindDates(rt)
        rt = clean(rt,
                fix_unicode=True,
                to_ascii=True,
                lower=True,
                normalize_whitespace=True,
                no_line_breaks=True,
                strip_lines=False,
                keep_two_line_breaks=False,
                no_urls=False,
                no_emails=False,
                no_phone_numbers=False,
                no_numbers=False,
                no_digits=False,
                no_currency_symbols=False,
                no_punct=False,
                no_emoji=True,
                replace_with_url="<URL>",
                replace_with_email="<EMAIL>",
                replace_with_phone_number="<PHONE>",
                replace_with_number="<NUMBER>",
                replace_with_digit="0",
                replace_with_currency_symbol="<CUR>",
                replace_with_punct=" ",
                lang="en"
                )
        rt = repl.replace_punct(rt)
        rt = FindDates(rt)
        new_data.append(rt)
        word_tokens = word_tokenize(rt)
        word_tokens = [w for w in word_tokens if w not in stop_words]
        for x in range(n+1):
            ngrams[x] += list(zipngram(word_tokens, n=x))

    return ngrams,new_data

tringrams,new_data = getBiGrams(data, n=3, dlen=20)
c = collections.Counter(tringrams[1])
c2 = collections.Counter(tringrams[2])

for i,e in enumerate(new_data):
    mnths = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november","december", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",]
    p = regex.compile(r"\L<name>", name=mnths)
    m = p.findall(e)
    print(m)
    if e.find(" nov ")!=-1:
        print(i)

print("categories {} ".format(categories))

FindDates( "  nov 05   nov 07 ")

list(datefinder.find_dates(data[6]['RESUME_TEXT'], source=True, index=True))
list(dateparser.search.search_dates(data[6]['RESUME_TEXT']))
list(datefinder.find_dates(new_data[6], source=True, index=True))



list(datefinder.find_dates("duration  jul 12   nov 13 team size", source=True, index=True, strict=False))
matchesdf = datefinder.find_dates("duration 8/8/2008 05th Nov   nov 07 team size  5", source=True, index=True)
matchesdf = list(matchesdf)
print(matchesdf)
for match in matchesdf:
    dparsed = myparse_timestamp(match[1], formats=dfmt)
    print(dparsed)

rt = clean(data[6]['RESUME_TEXT'],fix_unicode=True, to_ascii=True, lower=True,  normalize_whitespace=True,
   no_line_breaks=True, strip_lines=False, keep_two_line_breaks=False, no_urls=False,
   no_emails=False, no_phone_numbers=False, no_numbers=False, no_digits=False,
  no_currency_symbols=False, no_punct=True, no_emoji=True, lang="en")

repl = Replace()
rt = repl.replace_punct(rt)

#print( f" datefinder {list(datefinder.find_dates(rt, source=True, index=True, strict=False))} " )
print( "===============================================================")
print( f" dateparser {list(dateparser.search.search_dates(rt))} ")
list(dateparser.search.search_dates(rt))

import re
class DateExp:
    def __init__(self):
        MM = r"(0?[1-9]|1[0-2])"
        DELIMITER = r"([^\w\d\r\n:])"
        DD = r"(0?[1-9]|[12]\d|30|31)"
        YY = r"(\d{4}|\d{2})"
        CLOSEB = r")"
        CLOSEB = r"\b)"
        OPENB = r"("
        OPENB = r"(\b"
        YYYYMMDD = [
                    OPENB + DD + DELIMITER + MM + DELIMITER + YY + CLOSEB,
                    OPENB + MM + DELIMITER + DD + DELIMITER + YY + CLOSEB,
                    OPENB + YY + DELIMITER + DD + DELIMITER + MM + CLOSEB,
                    OPENB + YY + DELIMITER + MM + DELIMITER + DD + CLOSEB,
                    OPENB + MM  + DELIMITER + YY + CLOSEB,
               ]
        drexp = "|".join(YYYYMMDD)

        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November","December", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        bB =  "|".join(months)

        BBDDYY = [
                    OPENB + DD + DELIMITER + bB + DELIMITER + YY + CLOSEB,
                    OPENB + bB + DELIMITER + DD + DELIMITER + YY + CLOSEB,
                    OPENB + YY + DELIMITER + DD + DELIMITER + bB + CLOSEB,
                    OPENB + YY + DELIMITER + bB + DELIMITER + DD + CLOSEB,
                    OPENB + bB + DELIMITER + YY + CLOSEB,
                    OPENB + YY + bB +  DD + CLOSEB,
                    OPENB + DD + bB + YY + CLOSEB,
                    OPENB + bB + YY + CLOSEB,
               ]

        drexp_bb = "|".join( BBDDYY )
        self.regexp = drexp + "|" + drexp_bb
        self.comp_re = re.compile(self.regexp)
        self.MLIST = YYYYMMDD + BBDDYY

    def get_expr_list(self):
        return self.MLIST

    def get_date_re(self):
        return self.regexp

    def findall(self, text:str, ):
        return self.comp_re.findall(text)

mydateparser = DateExp()
re_str = mydateparser.get_date_re()
import re
matches = re.findall( mydateparser.get_date_re(), new_data[6], flags=re.IGNORECASE| re.DOTALL )
for m in matches:
    m = list(m)
    print(m)
    for i, mi in enumerate(m):
        if mi.strip() == "":
            continue
        #print(f" val {mi} " )
        #print(f" {i} {mydateparser.get_expr_list()[i]} " )
    print("".join(m))
