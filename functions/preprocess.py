#tokenizer
import os
import re

stopwords=open(os.path.join('/home/dhurba/Documents/smart_write/deployment/stopwords.txt'),encoding='utf').read().split('\n')
stp_words=[]
for word in stopwords:
    if word != '':
        new_word=word.strip()
    stp_words.append(new_word)
stp_words=list(set(stp_words))


def sentence_into_words(text):
    punctuations =set([',', ';', '?', '!', '—', '-', '.','—','/',':','–'])
    for punct in punctuations:
        text = ' '.join(text.split(punct))
        
    words=text.split()
    word_lst=[]
    for word in words:
        for ch in word:
            if ch=='।':
                word=word.replace('।', '')
        word_lst.append(word)

    for word in word_lst:
        if word=='':
            word_lst.remove('')


            
    return word_lst

def preprocess(text):  #test should be in list

    text=[text]

    remove_xao = lambda x: re.sub('\xa0','',x)
    text=map(remove_xao,text)

     #remove '\n'
    remove_n = lambda x: re.sub('\n','',x)
    text=map(remove_n,text)

    #remove '\u200d'
    remove_u = lambda x: re.sub('\u200d','',x)
    text=map(remove_u,text)

    #remove '\u200d'
    remove_uc = lambda x: re.sub('\u200c','',x)
    text=map(remove_uc,text)

    remove_alphanumeric=lambda x: re.sub('[a-zA-Z0-9०-९]*','',x)
    text=map(remove_alphanumeric,text)

    text=[p.translate(str.maketrans('','',"'‘’#$%^&*+\[](){}<>=_|`~")) for p in text]

    text=[p.translate(str.maketrans('','',"।,;?!—-.—/:–")) for p in text]

    text=[q.translate(str.maketrans('','','"“”')) for q in text]

    remove_whitespaces=lambda x: x.strip()   #to remove the leading and trailing whitespaces.
    text=map(remove_whitespaces,text)
    
    return ''.join(text)

def remove_stopwords(text):
    lst=sentence_into_words(text)
    new_lst=[]
    for word in lst:
        if word not in stp_words:
            new_lst.append(word)
    return ' '.join(new_lst)