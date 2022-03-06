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
spacy.cli.download("en_core_web_trf")
import hashlib
from copy import deepcopy
import string
import regex
import csv
import spacy
nlp = spacy.load("en_core_web_lg")
nlp = spacy.load("en_core_web_trf")

from gensim.models.word2vec import Text8Corpus
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from gensim.test.utils import datapath

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

FILE_PATH = 'C://Users//a_rathi//LocalDocuments//AI//res_parser//ALLRES.json'

def load_data():
    with open(FILE_PATH) as json_file:
        data = json.load(json_file)
    return data
data = load_data()

class DateExp:
    def __init__(self):
        MM = r"(0?[1-9]|1[0-2])"
        DELIMITER = r"([^\w\d\r\n:])"
        DELIMITER = r"([-\/: %]\s*)"
        DD = r"(0?[1-9]|[12]\d|30|31)"
        YY = r"(\d{2,4})"
        YYYY = r"(\d{4})"
        CLOSEB = r")"
        CLOSEB = r"\b)"
        OPENB = r"("
        OPENB = r"("
        YYYYMMDD = [
                    OPENB + r"?P<One>\b"   + DD + DELIMITER + MM + DELIMITER + YY + CLOSEB,
                    OPENB + r"?P<Two>\b"   + MM + DELIMITER + DD + DELIMITER + YY + CLOSEB,
                    OPENB + r"?P<Three>\b" + YYYY + DELIMITER + DD + DELIMITER + MM + CLOSEB,
                    OPENB + r"?P<Four>\b"  + YYYY + DELIMITER + MM + DELIMITER + DD + CLOSEB,
                    OPENB + r"?P<Five>\b"  + MM  + DELIMITER + YYYY + CLOSEB,
               ]
        drexp = "|".join(YYYYMMDD)

        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November","December", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        bB = r"(" + r"|".join(months) + r")"

        BBDDYY = [
                    OPENB + r"?P<Six>\b"     + DD + DELIMITER + bB + DELIMITER + YY + CLOSEB,
                    OPENB + r"?P<Seven>\b"   + bB + DELIMITER + DD + DELIMITER + YY + CLOSEB,
                    OPENB + r"?P<Eight>\b"   + YY + DELIMITER + DD + DELIMITER + bB + CLOSEB,
                    OPENB + r"?P<Nine>\b"    + YY + DELIMITER + bB + DELIMITER + DD + CLOSEB,
                    OPENB + r"?P<Ten>\b"     + bB + DELIMITER + YY + CLOSEB,
                    OPENB + r"?P<Eleven>\b"  + YYYY + bB +  DD + CLOSEB,
                    OPENB + r"?P<Twelve>\b"  + DD + bB + YY + CLOSEB,
                    OPENB + r"?P<Thirteen>\b" + bB + YY + CLOSEB,
               ]

        drexp_bb = "|".join( BBDDYY )
        self.regexp = drexp + "|" + drexp_bb
        self.comp_re = re.compile(self.regexp, flags=re.IGNORECASE)
        self.MLIST = YYYYMMDD + BBDDYY

    def get_expr_list(self):
        return self.MLIST

    def get_date_re(self):
        return self.regexp

    def get_comp_re(self):
        return self.comp_re

    def findall(self, text:str, ):
        return self.comp_re.findall(text)


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

class FindPatterns:
    def __init__(self):
        self.dates_dict = {}
        self.findemails_regex = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+' )
        self.email_dict = {}
        self.findurls_regex = re.compile(r'\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b')
        self.url_dict = {}

    def hash_string(self, input):
        byte_input = input.encode()
        hash_object = hashlib.sha256(byte_input)
        return hash_object.hexdigest()

    def FindEmails(self, text:str) :
        matchs = self.findemails_regex.finditer( text )
        list_match = []
        for m in matchs:
            e = m.group(0)
            h = "EMAIL" + self.hash_string(e)+  "EMAIL"
            s = m.span()
            d2 = { e: {"start":s[0],"end":s[1],"email":e,"hash":h } }
            self.email_dict[h] = d2
            list_match.append( (e, h) )
        for l in list_match:
            text = re.sub(l[0], l[1], text)
        return text

    def FindUrls(self, text:str ):
        matchs = self.findurls_regex.finditer(text)
        list_match = []
        for m in matchs:
            e = m.group(0)
            h = "URL" + self.hash_string(e)+  "URL"
            s = m.span()
            d2 = { e: {"start":s[0],"end":s[1],"url":e,"hash":h } }
            self.url_dict[h] = d2
            list_match.append( (e, h) )
        for l in list_match:
            text = re.sub(l[0], l[1], text)
        return text

    def FindDates(self, text:str):
        mydateparser = DateExp()
        re_comp = mydateparser.get_comp_re()
        matchs = re_comp.finditer(text)
        list_match = []
        for m in matchs:
            e = m.group(0)
            h = "DATE" + self.hash_string(e)+  "DATE"
            s = m.span()
            d2 = { e: {"start":s[0],"end":s[1],"date":e,"hash":h } }
            self.dates_dict[h] = d2
            list_match.append( (e, h) )
        for l in list_match:
            text = re.sub(l[0], l[1], text)
        return text

def zipngram(text_list,n=2):
  return zip(*[text_list[i:] for i in range(n)])

class Replace:
    def __init__(self, replace_with=" "):
        self.punc_dict = dict.fromkeys(string.punctuation, replace_with)
        self.punc_dict.pop('"')
        self.punc_dict.pop('$')
        self.punc_dict.pop('%')
        self.punc_dict.pop('_')
        self.punc_dict.pop('.')
        self.punc_trans = str.maketrans(self.punc_dict)
        self.dotre = re.compile(pattern="([a-zA-Z_]+)\.([ \n])" )
    def replace_punct(self, text ):
        """
        Replace punctuations from ``text`` with whitespaces (or other tokens).
        """
        text = self.dotre.sub(repl=r"\g<1> . ", string = text)
        return text.translate( self.punc_trans )

findpatterns = FindPatterns()

def getBiGrams(data, n, findpatterns, dlen = None):
    new_data = []
    new_text = ""
    I = 0
    if dlen is None:
        dlen = len(data) +1
    filtered_sentence = []
    word_tokens_full = []
    stop_words = set(stopwords.words('english'))
    repl = Replace()

    for element in data:
        I = I + 1
        if (I%100==0):
            print(round( 100*I/dlen,1) , end = ",")
        if I > dlen:
            break
        rt = element['RESUME_TEXT']
        rt = findpatterns.FindEmails(rt)
        rt = findpatterns.FindUrls(rt)
        rt = findpatterns.FindDates(rt)
        rt = clean(rt,
                fix_unicode=True,
                to_ascii=True,
                lower=True,
                normalize_whitespace=False,
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
        rt = findpatterns.FindDates(rt)
        new_data.append(rt)
        word_tokens = word_tokenize(rt)
        word_tokens = [w for w in word_tokens ]
        word_tokens_full += word_tokens

    return word_tokens_full,new_data

word_tokens, new_data = getBiGrams(data, n=3, findpatterns=findpatterns, dlen=None)

############################################################

def nGrams(word_tokens, n):
    ngrams = {}
    for x in range(n+1):
        ngrams[x] = []
    for x in range(n+1):
        if x==1:
            ngrams[x] = word_tokens
        ngrams[x] += list(zipngram(word_tokens, n=x))
    return ngrams


def savecsv(data, csv_file):
    with open(csv_file,'w',encoding='utf-8',newline='') as out:
        csv_out=csv.writer(out)
        l = ["name","num"]
        #csv_out.writerow( l )
        for row in data.items():
            row = list(row[0]) + [row[1]]
            csv_out.writerow(row)

def savelistcsv(data, csv_file):
    with open(csv_file,'w',encoding='utf-8',newline='') as out:
        for row in data:
            out.write(row)
            out.write("\n")

def save_lines(data, lines_file):
    with open(lines_file,'w',encoding='utf-8') as out:
        fsep = "="*80
        for doc in data:
            for line in doc.splitlines(True):
                out.write(line )
            out.write("\n" + fsep + "\n"  )

save_lines(data=new_data, lines_file="lines.txt")

def isValidPos(posword):
    if posword == "PROPN" or posword == "NOUN":
        return True
    return False

def ign_stop(wword_list, sline):
    pos_words = {}
    is_a = {}
    is_stop = {}
    nlist = []
    stop_words = frozenset(stopwords.words('english'))
    if sline:
        doc = nlp(sline)
        for token in doc:
            pos_words[token.text] =  token.pos_
            is_a[token.text] =  token.is_alpha
            is_stop[token.text] =  token.is_stop

    try:
        for wword in wword_list:
            if not sline:
                wiplist = str(wword).split("_")
                doc = nlp(" ".join(wiplist))
                for token in doc:
                    pos_words[token.text] =  token.pos_
                    is_a[token.text] =  token.is_alpha
                    is_stop[token.text] =  token.is_stop

            wword = str(wword)
            if wword.find("_") == -1:
                nlist.append(wword)
            else:
                if len(set(wiplist).intersection(stop_words)) == 0:
                    if all( [ isValidPos(pos_words[w]) for w in wiplist] ):
                        nlist.append(wword)
                else:
                    #print(f"bad phrase {wiplist}")
                    nlist.extend(wiplist)
    except:
        pass
        #print(f"tuple {wword}")
    return nlist




def cPhrases(wtoks, fname, ofname=None, min_count=5, threshold=5.5):
    # Create training corpus. Must be a sequence of sentences (e.g. an iterable or a generator).
    sentences = Text8Corpus('lines.txt')
    # Each sentence must be a list of string tokens:
    # Train a phrase model on our training corpus.

    stop_words = set(stopwords.words('english'))
    stop_words.update(ENGLISH_CONNECTOR_WORDS)
    phrase_model = Phrases(sentences, max_vocab_size = 50000000,
                           min_count = min_count,
                           threshold = threshold,
                           connector_words = frozenset(stop_words))
    # Apply the trained phrases model to a new, unseen sentence.
    pm_tok = phrase_model[wtoks]
    phrases_tok = ign_stop(pm_tok, None)
    if ofname:
        with open(ofname, 'w', encoding='utf-8') as ofile:
            with open(fname,'r', encoding='utf-8') as infile:
                for line in infile.readlines():
                    phrase_line = ign_stop( phrase_model[line.split()], line)
                    ofile.write( " ".join(phrase_line) )
                    ofile.write( "\n" )
        # Export the trained model = use less RAM, faster processing. Model updates no longer possible.
    frozen_model = phrase_model.freeze()

    # Save / load models.
    frozen_model.save("my_phrase_model.pkl")

    return phrases_tok


# Train a toy phrase model on our training corpus.
phrases_tok = cPhrases(word_tokens, "lines.txt",
                       ofname="linesout.txt",
                       min_count=10,
                       threshold=50.1)


onlyphrases_tok = [ str(t) for t in phrases_tok if str(t).find("_") != -1 ]

onlyphrases_tok[:10]

savelistcsv(data=onlyphrases_tok, csv_file="phrases.csv")



p_c = collections.Counter(onlyphrases_tok)
p_c.most_common(10)

tringrams = nGrams(word_tokens,3)
c = collections.Counter(tringrams[1])
c2 = collections.Counter(tringrams[2])
c3 = collections.Counter(tringrams[3])
c3.most_common(50)

savecsv(data=c, csv_file="1gram.csv")
savecsv(data=c2, csv_file="2gram.csv")
savecsv(data=c3, csv_file="3gram.csv")

import textacy
text = new_data[3]
meta = {}
gentities = []
for nd in new_data:
    doc = textacy.make_spacy_doc((nd, meta), lang="en_core_web_lg")
    entities = textacy.extract.basics.entities( doc )
    gentities += list(entities)

len(gentities)

gentities[:50]
