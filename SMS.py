from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from flask import render_template
from flasgger import Swagger
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
# sms text preprocessiing 
from spellchecker import SpellChecker
import nltk
import string
import pyarabic.trans
from pyarabic.araby import strip_tashkeel
from nltk import word_tokenize
from urlextract import URLExtract
import numpy as np, pandas as pd
def normalizeNumbers(sms): # convert arabic numbers into english
 return pyarabic.trans.normalize_digits(sms, source='all', out='west')
def FindUrls(sms):
    # findall() has been used
    # with valid conditions for urls in string
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    #regex = r"(?i)\b((?:https|http?://|www|\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))*([.a-z0-9]+)"
    smstext=str(sms)
    url = re.findall(regex, smstext)
    return [x[0] for x in url]
def checkUrlIfFound(txturl):
  extractor = URLExtract()
  if extractor.has_urls(txturl):
    return 1
  else:
    return 0
def getDomainExtension(sms):
  if ".com" in sms:
    return 1
  elif ".sa" in sms:
    return 2
  elif ".org" in sms:
    return 3
  elif ".edu" in sms:
    return 4
  else:
    return 0
def getHttp(sms):
  if "http" in sms:
    return 1
  elif "https" in sms:
    return 2
  else: return 0
#!pip install Phonenumbers
import re


def extractPhoneType(sms):
  s = ''.join(str(x) for x in sms)  
  number_list = re.findall(r'^\d{3,10}', s)
  if len(number_list)==0:
    return 0
  else:
    for n in number_list:
      
      if n.startswith("05") or n.startswith("966") or n.startswith("00966"): 
        return 1
      elif n.startswith("01") or (len(n)==3 and n.startswith("9")):
        return  2
      else: return 1
def extractPhone(sms):
  p=str(sms)
  number_list = re.findall(r'\d{10}|d{3}', p)
  return number_list
def checkPhoneIfFound(sms):
  p=str(sms)
  number_list = re.findall(r'\d{10}|d{3}', p)
  if len(number_list)==0:
    return 0
  else:
    return 1
spell = SpellChecker(language='ar')

nltk.download('punkt')
nltk.download('stopwords')# download stop word file using nltk library
stop = set(nltk.corpus.stopwords.words("arabic"))
arabicPunctuations = [".","`","؛","<",">","(",")","*","&","^","%","]","[",",","ـ","،","/",":","؟",".","'","{","}","~","|","!","”","…","“","–"] # defining customized punctuation marks
englishPunctuations = string.punctuation # importing English punctuation marks
englishPunctuations = [word.strip() for word in englishPunctuations] # converting the English punctuation from a string to array for processing
punctuationsList = arabicPunctuations + englishPunctuations # creating a list of all punctuation marks
# loading imojis file
with open('emojis.csv','r',encoding='utf-8') as f:
    lines = f.readlines()
    emojis_ar = {}
    for line in lines:
        line = line.strip('\n').split(';')
        emojis_ar.update({line[0].strip():line[1].strip()})

def removeWhiteSpace(sms):# remove extra spaces
  newSms=sms.strip()
  return newSms
#Removing Punctuations


def removingPunctuation(text):
  cleanSMS = ''
  for i in text:
    if i not in punctuationsList:
      cleanSMS = cleanSMS + '' + i 
    else: cleanSMS = cleanSMS + ' '
  return cleanSMS



def removeEnglishCharacters(sent):
    """clean data from any English char, emoticons, underscore, and repeated > 2
    str -> str"""
    p1 = re.compile('\W')
    p2 = re.compile('\s+')
    sent = sent.replace('_', ' ')
    sent = re.sub(r'[A-Za-z0-9]', r'', sent)
    sent = re.sub(p1, ' ', sent)
    sent = re.sub(p2, ' ', sent)
    return sent
def removeNumbers(text): 
    # Remove numbers
    text = re.sub("\d+", " ", text)
    return text


# analysis precentage error of spilling and grammer of sms
def percenatgeErrorsPerSms(sms):
    NumberOfWords=0
    txt=word_tokenize(sms)
    print(txt)
    numberOfMistakes=0.0
    count=0
    precenatgeError=0.0
    
    
    NumberOfWords=len(txt)
    for w1 in txt:
      w2=spell.correction(w1)
      #print(w2)
      if(w1!=w2):
        count=count+1
    if NumberOfWords==0:
      return precenatgeError
    precenatgeError=count/NumberOfWords
    
    print(precenatgeError)
    return precenatgeError
#Stop Words Removal

def stopRemoval(sms):
    temptweet = word_tokenize(sms)
    sms= " ".join([w for w in temptweet if not w in stop and len(w) >= 2])
    return sms
def norm_alif(text):
    text = text.replace(u"\u0625", u"\u0627")  # HAMZA below, with LETTER ALEF
    #text = text.replace(u"\u0621", u"\u0627")  # HAMZA, with LETTER ALEF
    text = text.replace(u"\u0622", u"\u0627")  # ALEF WITH MADDA ABOVE, with LETTER ALEF
    text = text.replace(u"\u0623", u"\u0627")  # ALEF WITH HAMZA ABOVE, with LETTER ALEF
    return text
def norm_taa(text):
    text=text.replace(u"\u0629", u"\u0647") # taa' marbuuTa, with haa'
    #text=text.replace(u"\u064A", u"\u0649") # yaa' with 'alif maqSuura
    return text
def norm_yaa(text):
    if len(text)!=0:
        if text[-1] == u"\u064A":
            text = text.replace(u"\u064A", u"\u0649")  # yaa' with 'alif maqSuura
    return text
def eliminate_single_char_words(Tweet):
    parts = Tweet.split(" ")
    cleaned_line_parts = []
    for P in parts:
        if len(P) != 1:
            cleaned_line_parts.append(P)
    cleaned_sms = ' '.join(cleaned_line_parts)
    return cleaned_sms
#========================================================Converting Emoji



def remove_emoji(text):
    emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text



def emoji_native_translation(text):
    text = text.lower()
    loves = ["<3", "♥",'❤']
    smilefaces = []
    sadfaces = []
    neutralfaces = []

    eyes = ["8",":","=",";"]
    nose = ["'","`","-",r"\\"]
    for e in eyes:
        for n in nose:
            for s in ["\)", "d", "]", "}","p"]:
                smilefaces.append(e+n+s)
                smilefaces.append(e+s)
            for s in ["\(", "\[", "{"]:
                sadfaces.append(e+n+s)
                sadfaces.append(e+s)
            for s in ["\|", "\/", r"\\"]:
                neutralfaces.append(e+n+s)
                neutralfaces.append(e+s)
            #reversed
            for s in ["\(", "\[", "{"]:
                smilefaces.append(s+n+e)
                smilefaces.append(s+e)
            for s in ["\)", "\]", "}"]:
                sadfaces.append(s+n+e)
                sadfaces.append(s+e)
            for s in ["\|", "\/", r"\\"]:
                neutralfaces.append(s+n+e)
                neutralfaces.append(s+e)

    smilefaces = list(set(smilefaces))
    sadfaces = list(set(sadfaces))
    neutralfaces = list(set(neutralfaces))
    t = []
    for w in text.split():
        if w in loves:
            t.append("حب")
        elif w in smilefaces:
            t.append("مضحك")
        elif w in neutralfaces:
            t.append("عادي")
        elif w in sadfaces:
            t.append("محزن")
        else:
            t.append(w)
    newText = " ".join(t)
    return newText

def is_emoji(word):
    if word in emojis_ar:
        return True
    else:
        return False
    
def add_space(text):
    return ''.join(' ' + char if is_emoji(char) else char for char in text).strip()

def translate_emojis(words):
    word_list = list()
    words_to_translate = list()
    for word in words :
        t = emojis_ar.get(word.get('emoji'),None)
        if t is None:
            word.update({'translation':'عادي','translated':True})
            #words_to_translate.append('normal')
        else:
            word.update({'translated':False,'translation':t})
            words_to_translate.append(t.replace(':','').replace('_',' '))
        word_list.append(word)
    return word_list

def emoji_unicode_translation(text):
    text = add_space(text)
    words = text.split()
    text_list = list()
    emojis_list = list()
    c = 0
    for word in words:
        if is_emoji(word):
            emojis_list.append({'emoji':word,'emplacement':c})
        else:
            text_list.append(word)
        c+=1
    emojis_translated = translate_emojis(emojis_list)
    for em in emojis_translated:
        text_list.insert(em.get('emplacement'),em.get('translation'))
    text = " ".join(text_list)
    return text
    
def clean_emoji(text):
    text = emoji_native_translation(text)
    text = emoji_unicode_translation(text)
    return text

def normalize_numbers(text):
    # TODO: Implement this function to normalize numbers in text
    return text

def strip_tashkeel(text):
    # TODO: Implement this function to strip tashkeel marks from text
    return text

model = load_model('arabert_model.h5')

maxlen =200
def preprocess_text(sms):
  tokenizer = Tokenizer()
  with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
  max_features = len(tokenizer.word_index) + 1
  print(max_features)
  smstoken= tokenizer.texts_to_sequences([sms])
  # list_tokenized_test = tokenizer.texts_to_sequences(testsms)

  processed_sms = pad_sequences(smstoken, maxlen=maxlen)
  return processed_sms

def extract_features(text):
    # TODO: Implement this function to extract additional features from the text
    return np.zeros(6)

def preprocessing(text):
    urlText = str(FindUrls(text))
    urlFound=checkUrlIfFound(urlText )
    httpType=getHttp(text)
    extensionDomain=getDomainExtension(text)
    PhoneNo=extractPhone(text)
    containPhoneNo=checkPhoneIfFound(text)
    PhoneType=extractPhoneType(PhoneNo)
    text = removeWhiteSpace(text)
    text = clean_emoji(text)
    text= removingPunctuation(text)
    text= removeEnglishCharacters(text)
    text= removeNumbers(text)
    precenatgeError = percenatgeErrorsPerSms(text)
    text= stopRemoval(text)
    text= norm_alif(text)
    text= norm_taa(text)
    text = norm_yaa(text)
    text = eliminate_single_char_words(text)

    return [text,urlFound , precenatgeError,containPhoneNo, extensionDomain,httpType,PhoneType]      

# Define the API endpoint for receiving SMS messages
app = Flask(__name__)
run_with_ngrok(app)
Swagger(app)
from pyngrok import ngrok
ngrok.set_auth_token('2LzEGKEZyOxWD3tLDcK6YPxlSC8_2cHXfQ6K3syud3kBM8EaW')
@app.route('/sms', methods=['POST'])
def receive_sms():
    input_text = request.form.get('Body')
    if input_text is None:
        return jsonify({'error': 'No input text provided'})
    
    data = preprocessing(input_text)
   
    sms=data[0]
    maxlen =200
    tokenizer = Tokenizer()
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    max_features = len(tokenizer.word_index) + 1
    print(max_features)
    smstoken= tokenizer.texts_to_sequences([sms])
    X_new = pad_sequences(smstoken, maxlen=maxlen)
    print(X_new.shape)
    aux=np.array(data[1:])
    additional_features = np.array(aux).reshape(1,-1)
    # additional_features = np.expand_dims(additional_features, axis=0)
    data[1:]
    prediction = model.predict([X_new, additional_features])
    if(prediction>0.5):
      prediction='SPAM'
    else:
      prediction='Not SPAM'
    response = {'prediction': prediction}

    return json.dumps(response)
    response = {'prediction': prediction}
    return json.dumps(response)

# Define a welcome message for the root endpoint
@app.route('/')
def welcome():
     return render_template('index.html')

if __name__ == '__main__':
    app.run()
