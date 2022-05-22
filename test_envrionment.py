import chunk
import csv
from turtle import shape
from itsdangerous import json
from nbformat import read
from pyLDAvis import js_PCoA
import unidecode

### os utility
import os
os.listdir("Y:\\data\\00000000")
# Limited by missing wildcards in path

#-------------------------------------------#

### WSL subprocess
#e.g: To execute "ls -l"
import subprocess
command=["wsl", "cd","~/../../mnt/share/"]
command=["wsl", "pwd"]
print(subprocess.check_call(command))
# Complex interaction


### glob
import glob
#Load list of filenames
dir=str("00000000\ ")
prefix=r"Y:\data\ " + dir
prefix=prefix.replace(" ","")
targetPattern = prefix +"**\*.txt"
testList=glob.glob(targetPattern)


# Create a dictionary with the doi and path separated

# example
testString=testList[0]
testString=testString.replace(".txt","")
rmString=prefix
testString=testString.replace(rmString,"")
testString=testString.replace("\\","/")
testString
from urllib.parse import unquote
doi = unquote(testString)
print(doi)
# Print corresponding data
with open(testList[0], "r", encoding="utf8") as f:
    contents = f.read()
    len(contents)
    print(contents[0:100])



# Dictionary with path and doi name
dictDoiPath={}

for path in testList:
    testString=path.replace(".txt","")
    rmString=prefix
    testString=testString.replace(rmString,"")
    testString=testString.replace("\\","/")
    testString
    from urllib.parse import unquote
    doi = unquote(testString)
    dictDoiPath[doi]=path

len(dictDoiPath)

import random
# Get random dictionary pair in dictionary
# Using random.choice() + list() + items()
res = key, val = random.choice(list(dictDoiPath.items()))
# printing result
print("The random pair is : " + str(res))

# Print corresponding data
with open(val, "r", encoding="utf8") as f:
    contents = f.read()
    len(contents)
    print(contents[0:100])

# Multicore processing and paralleization
#-------------------------------------------#

# Load packages
from Functions.F1_Subsets_and_PreProcessing import DOI_Path_Dictionary, Random_DOI_Path_Pair, Preprocess_Token_List
import pickle
import pandas as pd
import glob
from urllib.parse import unquote
import random
import nltk
import re
import string
import gensim
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer # or LancasterStemmer, RegexpStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer 
# nltk.download('punkt')
# nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = stopwords.words('english') # or any other list of your choice
lemmatizer = WordNetLemmatizer()
# pip install langdetect
from langdetect import detect_langs
import time

# Define input paths and file names
doiPath_Path="Y:\DoiPathDicts\ "
doiPath_Suffix="_DOI_Path_Dict.pkl"
# Define Output paths and file names
fullTextDf_Path="Y:\IntermediateData\ "

# Set the dirs to save doi and paths
StartDir=0
EndDir=1

dictList=[]
from mp_functions import Doi_Path_Dict
# Iterate trough data directories
for dirNum in range(StartDir,EndDir+1):
    interDict=Doi_Path_Dict(dirNum, doiPath_Path, doiPath_Suffix)      
    dictList.append(interDict)

# Check if dictionaries are different
for dictItem in dictList:
    print(len(dictItem),list(dictItem.keys())[0:2])

# Maybe split one dictionary into subdictionaries
# # # # # # # # # # # # # # # # # # # # # # # # # # 

from itertools import islice

def chunks(data, SIZE):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}

slicedDictList=[]
for item in chunks(dictList[0], 10000):
    slicedDictList.append(item)

for item in slicedDictList:
    print(len(item))

# # # # # # # # # # # # # # # # # # # # # # # # # # 


import itertools
testDict1=dict(itertools.islice(dictList[0].items(), 50000))
testDict2=dict(itertools.islice(dictList[1].items(), 50000))

# Time 100 Fulltext Documents

# 10000 Docs
# Time elapsed in secdongs:  1067.4442416 , in minutes  17.790737359999998 , in hours:  0.2965122893333333
from mp_functions import Preprocessed_Dict_and_Metadata
test=Preprocessed_Dict_and_Metadata(testDict1)
test=Preprocessed_Dict_and_Metadata(testDict2)

# 10 Processes: 
import multiprocessing as mp
import sys, importlib
importlib.reload(sys.modules['mp_functions'])
from mp_functions import Preprocessed_Dict_and_Metadata
pool = mp.Pool(processes=10)
# Return=pool.map(Preprocessed_Dict_and_Metadata, [testDict1,testDict2])
# Return=pool.map(Preprocessed_Dict_and_Metadata, [dictList[0],dictList[1]])
Return=pool.map(Preprocessed_Dict_and_Metadata, slicedDictList)
pool.close
# Check that each dict is processed separately
for dataSet in Return:
    print(len(dataSet),dataSet[0].head(3))

# Inspect the amount of encodingErrors
for dataSet in Return:
    print(len(dataSet[2]))

# Append Metadata
slicedMetaDataList=[]
for item in Return:
    type(item[0])
    slicedMetaDataList.append(item[0])
len(slicedMetaDataList)
metaData = pd.concat(slicedMetaDataList)
len(metaData)
metaData.iloc[[50000]]

# Append Dictionaries
FtPr={}
for item in Return:
    type(item[1])
    FtPr.update(item[1])
len(FtPr)
FtPr[metaData.iloc[[50000]]["DOI"][0]][0:100]


len(Return[0][0])
len(Return[0][1])
len(Return[1][0]) 
len(Return[1][1])
len(Return[2][0])
len(Return[2][1])
len(Return[3][0])
len(Return[3][1])


# Multicore processing and paralleization 2
#-------------------------------------------#

# Load packages
from Functions.F1_Subsets_and_PreProcessing import DOI_Path_Dictionary, Random_DOI_Path_Pair, Preprocess_Token_List
from Functions.F1_Subsets_and_PreProcessing import Preprocessed_Dict_and_Metadata, Doi_Path_Dict_Loader, Chunks
import pickle
import pandas as pd
import glob
from urllib.parse import unquote
import random
import nltk
import re
import string
import gensim
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer # or LancasterStemmer, RegexpStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer 
# nltk.download('punkt')
# nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = stopwords.words('english') # or any other list of your choice
lemmatizer = WordNetLemmatizer()
# pip install langdetect
from langdetect import detect_langs
import time
import multiprocessing as mp
from itertools import islice

# Define input paths and file names
doiPath_Path="Y:\DoiPathDicts\ "
doiPath_Suffix="_DOI_Path_Dict.pkl"

# Define Output paths and file names
IntermediateData_Path="Y:\IntermediateData\ "
FtPr_Suffix="_FtPr.pkl"
MetaData_Suffix="_MetaData.pkl"
encodeError_Suffix="_MetaData.pkl"

# Set the dirs to save doi and paths
StartDir=0
EndDir=1

# Iterate trough data directories
for dirNum in range(StartDir,EndDir+1):
    #Load the dict
    dictItem=Doi_Path_Dict_Loader(dirNum, doiPath_Path, doiPath_Suffix)  
    print(len(dictItem),list(dictItem.keys())[0:2])

    # Init a list which slices the dictionary into multiple dictionaries (each a chunk af 10000)
    slicedDictList=[]
    # Create and append dictionary chunks
    for item in Chunks(dictItem, 10000):
        slicedDictList.append(item)
        print(len(item))

    # Process each dictionary chunk
    print("Available cores: ",mp.cpu_count(), "(Pool = amount of cores)")
    pool = mp.Pool(processes=10)
    print("pool with 10 processes")
    Return=pool.map(Preprocessed_Dict_and_Metadata, slicedDictList)
    pool.close


    # Append Metadata
    slicedMetaDataList=[]
    for item in Return:
        slicedMetaDataList.append(item[0])
    metaData = pd.concat(slicedMetaDataList)
    print("Apppended all of the metaData for file: ", dirNum)

    # Append preprocessed text Dictionaries
    FtPr={}
    for item in Return:
        FtPr.update(item[1])
    print("Apppended all of the FtPr dictionaries for file: ", dirNum)

    # Append encoidng error dicitonaries
    encErr={}
    for item in Return:
        encErr.update(item[2])
    print("Apppended all of the encoding error dictionaries for file: ", dirNum)

    # Create name for the metaData
    # Bring for example 27 into the form of "027"
    dirNum=str(dirNum).zfill(3)
    # Create path to dictionary
    metaDataName=(IntermediateData_Path + dirNum + MetaData_Suffix).replace(" ","")
    FtPrName=(IntermediateData_Path + dirNum + MetaData_Suffix).replace(" ","")
    encErrName=(IntermediateData_Path + dirNum + encodeError_Suffix).replace(" ","")


    # Store the returned elements
    # create a binary pickle file 
    a = open(FtPrName,"wb")
    # write the python object (dict) to pickle file
    pickle.dump(FtPr,a)
    # close file
    a.close()

    # Store the returned elements
    # create a binary pickle file 
    b = open(encErrName,"wb")
    # write the python object (dict) to pickle file
    pickle.dump(encErr,b)
    # close file
    b.close()

    # Save dataframe of metaData
    metaData.to_pickle(metaDataName)




# Meta Data of preprocessing
#-------------------------------------------#
# Load packages
# from Functions.F1_Subsets_and_PreProcessing import DOI_Path_Dictionary, 
# Random_DOI_Path_Pair, 
# Preprocess_Token_List
from Functions.F1_Subsets_and_PreProcessing import Dict_Loader
# Preprocessed_Dict_and_Metadata, 
# , 
# Chunks
# import pickle
import pandas as pd
# import glob
# from urllib.parse import unquote
# import random
# import nltk
# import re
# import string
# import gensim
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer # or LancasterStemmer, RegexpStemmer, SnowballStemmer
# from nltk.stem import WordNetLemmatizer 
# # nltk.download('punkt')
# # nltk.download('stopwords')
# stemmer = PorterStemmer()
# stop_words = stopwords.words('english') # or any other list of your choice
# lemmatizer = WordNetLemmatizer()
# # pip install langdetect
# from langdetect import detect_langs
# import time
# import multiprocessing as mp
# from itertools import islice
import matplotlib.pyplot as plt


# Define input paths and file names
IntermediateData_Path="Y:\IntermediateData\ "
FtPr_Suffix="_FtPr.pkl"
MetaData_Suffix="_MetaData.pkl"
# encodeError_Suffix="_MetaData.pkl"

# # Set the dirs to save doi and paths
# StartDir=0
# EndDir=1

# # Iterate trough data directories
# for dirNum in range(StartDir,EndDir+1):
#     #Load the dict with processed texts

dirNum=0
interDict=Dict_Loader(dirNum, IntermediateData_Path, FtPr_Suffix)


dirNum=str(dirNum).zfill(3)
metaDataName=(IntermediateData_Path + dirNum + MetaData_Suffix).replace(" ","")
metaData=pd.read_pickle(metaDataName)
metaData.columns

metaData['Token Amount'].plot()

metaData['Token Amount'].sort_values(ascending=True).head(10)
metaData['Token Amount'].isnull().sum().sum()
type(metaData['Token Amount'])
metaData['Token Amount'].sort_values(ascending=False).head(10)
metaData['DOI'][metaData['Token Amount']==114967].item()


pd.DataFrame(metaData['Token Amount']==0).value_counts()
pd.DataFrame((metaData['Token Amount']<26617) & (metaData['Token Amount']>50)).value_counts()

metaData['Token Amount'][(metaData['Token Amount']<26617) & (metaData['Token Amount']>50)].plot.hist(bins=1000,density=True)
plt.show()

metaData['Token Amount'][metaData['Token Amount']>50]







# Test log functions
#-------------------------------------------------------
# Load packages
# from Functions.F1_Subsets_and_PreProcessing import DOI_Path_Dictionary, Random_DOI_Path_Pair #, Preprocess_Token_List
from Functions.F1_Subsets_and_PreProcessing import  Dict_Loader #, Preprocessed_Dict_and_Metadata, Chunks
import pickle
import pandas as pd
# import glob
# from urllib.parse import unquote
# import random
# import nltk
# import re
# import string
import gensim
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer # or LancasterStemmer, RegexpStemmer, SnowballStemmer
# from nltk.stem import WordNetLemmatizer 
# nltk.download('punkt')
# nltk.download('stopwords')
# stemmer = PorterStemmer()
# stop_words = stopwords.words('english') # or any other list of your choice
# lemmatizer = WordNetLemmatizer()
# pip install langdetect
# from langdetect import detect_langs
import time
# import multiprocessing as mp
# from itertools import islice
# import pyLDAvis.gensim_models
import gc
import memory_profiler as mem_profile




# Define input paths and file names
IntermediateData_Path="Y:\\IntermediateData\\"
FtPr_Suffix="_FtPr.pkl"
MetaData_Suffix="_MetaData.pkl"
# Define output path and file names
Model_Path="Y:\\Models\\"
# Phrases Bigram
FreezedPhrases_Suffix="_FreezedBigram.pkl"
FreezedPhrasesLog_Suffix="_FreezedBigramLog.txt"
# Filtered meta data
MetaDataFiltered_Suffix="_MetaDataFiltered.pkl"
# Define Bigram/Phrase Parameters
bigramMinFreq=10
bigramThreshold=10
phraseVocabSize=400000000
# Set the dirs to save doi and paths
StartDir=0
EndDir=1


def path_creator(pathType, ArgumentList):
    if pathType=="log" or pathType=="phraseModel":
        return (ArgumentList[0] + str(ArgumentList[1]).zfill(3) + "_" + str(ArgumentList[2]).zfill(3) + ArgumentList[3])
    
    elif pathType=="meta":
        return (ArgumentList[0] + str(ArgumentList[1]).zfill(3) + ArgumentList[2])


def log_printer(logName, ArgumentList):
    if ArgumentList[0]=="PhraseModelParameters":
        print("Phrase Model Parameters: ", ArgumentList[1], ArgumentList[2], ArgumentList[3],file=open(logName,'a'))
        print("Phrase Model Parameters: ", ArgumentList[1], ArgumentList[2], ArgumentList[3])

    elif ArgumentList[0]=="MemoryUsage":
        print(ArgumentList[1], "{} Mb".format(mem_profile.memory_usage()),file=open(logName,'a'))
        print(ArgumentList[1], "{} Mb".format(mem_profile.memory_usage()))

    elif ArgumentList[0]=="ProcessTime":
        print(ArgumentList[1], ArgumentList[2], " is loaded/processed... in seconds: ", round(ArgumentList[3],4), ", in minutes ", round((ArgumentList[3])/60,4), ", in hours: ", round((ArgumentList[3])/3600,4),file=open(logName,'a'))
        print(ArgumentList[1], ArgumentList[2], " is loaded/processed... in seconds: ", round(ArgumentList[3],4), ", in minutes ", round((ArgumentList[3])/60,4), ", in hours: ", round((ArgumentList[3])/3600,4))

    elif ArgumentList[0]=="MetaFilter":
        print("Num of Docs before/after filtering: ", ArgumentList[1], " / ", ArgumentList[2],file=open(logName,'a'))
        print("Num of Docs before/after filtering: ", ArgumentList[1], " / ", ArgumentList[2])

    elif ArgumentList[0]=="BigramUpdate":
        print("Updated Bigram Vocab is: ",ArgumentList[1],file=open(logName,'a'))
        print("Updated Bigram Vocab is: ",ArgumentList[1])

    elif ArgumentList[0]=="iterNext":
        print("-------------------------------------------------------",file=open(logName,'a'))
        print("-------------------------------------------------------")

    else:
        print(ArgumentList[0], " Argument is not defined")
    
    # #---------------------------------------------------------------------------------------------------------------
    # # Check if new vocab is added through the counting of bigrams (If vocab is updated it shoud be the same)
    # AllBigrams=[len([b for b in bigram[sent] if b.count('_') == 1]) for sent in FtPrFi[0:10000]]

    # print("Bigram count of ", dirNum, " is ", sum(AllBigrams), "(Calculated for 10000 docs at each iteration)",file=open(saveName,'a'))
    # print("Bigram count of ", dirNum, " is ", sum(AllBigrams), "(Calculated for 10000 docs at each iteration)")

    # # Check if new vocab is added through the phasegrams dict of the freezed model
    # bigram_freezed = bigram.freeze()

    # print("Amount of phrasegrams in freezed bigram model: ",len(bigram_freezed.phrasegrams),file=open(saveName,'a'))
    # print("-------------------------------------------------------",file=open(saveName,'a'))
    # print("Amount of phrasegrams in freezed bigram model: ",len(bigram_freezed.phrasegrams))   




# Start Global Timer which times the whole function
tic = time.perf_counter()

# Initiate Bigram Model
bigram = gensim.models.phrases.Phrases(min_count=bigramMinFreq, threshold=bigramThreshold, max_vocab_size=phraseVocabSize)

# Print parameters of phrase model
log_printer(path_creator("log",[Model_Path, StartDir, EndDir, FreezedPhrasesLog_Suffix]),["PhraseModelParameters", bigramMinFreq, bigramThreshold, phraseVocabSize])

# Print Memory usage at start
log_printer(path_creator("log",[Model_Path, StartDir, EndDir, FreezedPhrasesLog_Suffix]),["MemoryUsage", "Usage before processing: "])

# Create phrases model
for dirNum in range(StartDir,EndDir+1):

    # Load FilteredMetaData
    Meta=pd.read_pickle(path_creator("meta",[IntermediateData_Path, dirNum, MetaDataFiltered_Suffix]))

    #Load preprocessed full text
    tictic = time.perf_counter()
    FtPr=Dict_Loader(dirNum, IntermediateData_Path, FtPr_Suffix)
    toctoc = time.perf_counter()
    #Print time usage to log
    log_printer(path_creator("log",[Model_Path, StartDir, EndDir, FreezedPhrasesLog_Suffix]),["ProcessTime","Data of directory: ",dirNum,toctoc-tictic])

    # Apply meta data filter and delet old dataframe
    FtPrFi=[FtPr[DOI] for DOI in list(Meta["DOI"])]
    # Check if apllying the filter worked
    log_printer(path_creator("log",[Model_Path, StartDir, EndDir, FreezedPhrasesLog_Suffix]),["MetaFilter",len(FtPr),len(FtPrFi)])

    # Print Memory usage at start
    log_printer(path_creator("log",[Model_Path, StartDir, EndDir, FreezedPhrasesLog_Suffix]),["MemoryUsage", "Usage before deleting FtPr: "])
    del FtPr
    gc.collect
    # Print Memory usage at start
    log_printer(path_creator("log",[Model_Path, StartDir, EndDir, FreezedPhrasesLog_Suffix]),["MemoryUsage", "Usage after deleting FtPr: "])

    #Load preprocessed full text
    tictic = time.perf_counter(
    # Update the Phrase Model
    bigram.add_vocab(FtPrFi)
    toctoc = time.perf_counter()
    #Print time usage to log
    log_printer(path_creator("log",[Model_Path, StartDir, EndDir, FreezedPhrasesLog_Suffix]),["ProcessTime","Updating of biram model with directory: ",dirNum,toctoc-tictic])
    # log updated bigram
    log_printer(path_creator("log",[Model_Path, StartDir, EndDir, FreezedPhrasesLog_Suffix]),["BigramUpdate",len(bigram.vocab)])


    # Print Memory usage at start
    log_printer(path_creator("log",[Model_Path, StartDir, EndDir, FreezedPhrasesLog_Suffix]),["MemoryUsage", "Usage before deleting FtPrFi: "])
    # Delete loaded lists
    del(FtPrFi)
    gc.collect()
    # Print Memory usage at start
    log_printer(path_creator("log",[Model_Path, StartDir, EndDir, FreezedPhrasesLog_Suffix]),["MemoryUsage", "Usage after deleting FtPrFi: "])

    # Print iteration separator
    log_printer(path_creator("iterNext",[Model_Path, StartDir, EndDir, FreezedPhrasesLog_Suffix]),[])

# Freeze Model
frozen_bigram_model = bigram.freeze()

# Save Model
frozen_bigram_model.save(path_creator("phraseModel",[Model_Path, StartDir, EndDir, FreezedPhrases_Suffix]))

#Stop global timer and print
toc = time.perf_counter()
log_printer(path_creator("log",[Model_Path, StartDir, EndDir, FreezedPhrasesLog_Suffix]),["ProcessTime","Updating of biram model with directory: ",dirNum,toc-tic])








# If conda is not activated, invoke python then hit ctrl+Z then type conda activate to activate conda

# Try applying generators for data streaming
#-------------------------------------------------------
# Load packages
from Functions.F1_Subsets_and_PreProcessing import Dict_Loader
from Functions.F2_Model_Building import path_creator, log_printer
import pickle
import pandas as pd
import gensim
import time
# import pyLDAvis.gensim_models
import gc
import memory_profiler as mem_profile

# Define input paths and file names
IntermediateData_Path="Y:\\IntermediateData\\"
FtPr_Suffix="_FtPr.pkl"
MetaData_Suffix="_MetaData.pkl"
# Define output path and file names
Model_Path="Y:\\Models\\"
Dictionary_Suffix="_Dictionary"
BowCorpus_Suffix="_BowCorpus.mm"
# Define Dictionary Parameters
dictVocab=60000000
filter_freq=10
# Define LDA Parameters
nTopics=50
# Set the dirs to save doi and paths
StartDir=0
EndDir=49



# class MyCorpus:
#     def __iter__(self):
#         for line in open('https://radimrehurek.com/mycorpus.txt'):
#             # assume there's one document per line, tokens separated by whitespace
#             yield dictionary.doc2bow(line.lower().split())


# Load dictionary
dct=gensim.corpora.Dictionary.load(path_creator("dictionary",[Model_Path, StartDir, EndDir, Dictionary_Suffix]))
# Remove words with frequency lower than 2
dct.filter_extremes(no_below=filter_freq, keep_n=None)
# Initiate LDA Model
ldaModel= gensim.models.ldamodel.LdaModel(id2word=dct, num_topics=nTopics)


# dirNum=1
# def corpusStream(IntermediateData_Path, dirNum, StartDir, EndDir, BowCorpus_Suffix):
#     for doc in gensim.corpora.MmCorpus(path_creator("bowCorpus",[IntermediateData_Path, dirNum, StartDir, EndDir, BowCorpus_Suffix])):
#         yield doc
# test=corpusStream(IntermediateData_Path, dirNum, StartDir, EndDir, BowCorpus_Suffix)
# test=list(test)

ldaModel= gensim.models.ldamodel.LdaModel(id2word=dct, num_topics=nTopics)
dirNum=1
"{} Mb".format(mem_profile.memory_usage())
ldaModel.update([doc for doc in gensim.corpora.MmCorpus(path_creator("bowCorpus",[IntermediateData_Path, dirNum, StartDir, EndDir, BowCorpus_Suffix]))])
"{} Mb".format(mem_profile.memory_usage())

ldaModel= gensim.models.ldamodel.LdaModel(id2word=dct, num_topics=nTopics)
"{} Mb".format(mem_profile.memory_usage())
corpus=gensim.corpora.MmCorpus(path_creator("bowCorpus",[IntermediateData_Path, dirNum, StartDir, EndDir, BowCorpus_Suffix])
for doc in corpus:
    ldaModel.update(doc)
"{} Mb".format(mem_profile.memory_usage())





# Load Corpus into memory
"{} Mb".format(mem_profile.memory_usage())
# Initiate LDA Model
ldaModel= gensim.models.ldamodel.LdaModel(id2word=dct, num_topics=nTopics)
dirNum=1
bowCorpus=gensim.corpora.MmCorpus(path_creator("bowCorpus",[IntermediateData_Path, dirNum, StartDir, EndDir, BowCorpus_Suffix]))
ldaModel.update(corpusStream(IntermediateData_Path, dirNum, StartDir, EndDir, BowCorpus_Suffix))
"{} Mb".format(mem_profile.memory_usage())



# # gensim stream of documents
# # corpus = MyCorpus()
# def corpus_stream(IntermediateData_Path, dirNum, StartDir, EndDir, FtPr_Suffix):
#     FtPrFi=pickle.load(open(path_creator("FtPr",[IntermediateData_Path, dirNum, StartDir, EndDir, FtPr_Suffix]), "rb"))
#     for doc in FtPrFi:
#         yield dct.doc2bow(doc)

# ldaModel.update(corpus_stream(IntermediateData_Path, dirNum, StartDir, EndDir, FtPr_Suffix))

# # ldaModel.update(corpus)






# js_PCoA
#-------------------------------------------------------
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import gensim
import numpy as np
from Functions.F2_Model_Building import path_creator
IntermediateData_Path="Y:\\IntermediateData\\"
Model_Path="Y:\\Models\\"
MetaDataFiltered_Suffix="_MetaDataFiltered.pkl"
Dictionary_Suffix="_Dictionary"
LDA_Suffix="_LDA.model"
dictVocab=60000000
filter_freq=10
# Define LDA Parameters
nTopics=50
# Set the dirs to save doi and paths
StartDir=0
EndDir=49

# Load model
topic_model =  gensim.models.LdaModel.load(path_creator("ldaModel",[Model_Path, StartDir, EndDir, LDA_Suffix]))

# Load dictionary
dictionary=gensim.corpora.Dictionary.load(path_creator("dictionary",[Model_Path, StartDir, EndDir, Dictionary_Suffix]))

# Remove words with frequency lower than 2
dictionary.filter_extremes(no_below=filter_freq, keep_n=None)

# Get list of dictionary ids
fnames_argsort = np.asarray(list(dictionary.token2id.values()), dtype=np.int_)


topic = topic_model.state.get_lambda()
topic = topic / topic.sum(axis=1)[:, None]
topic_term_dists = topic[:, fnames_argsort]


def js_PCoA(distributions):
    """Dimension reduction via Jensen-Shannon Divergence & Principal Coordinate Analysis
    (aka Classical Multidimensional Scaling)
    Parameters
    ----------
    distributions : array-like, shape (`n_dists`, `k`)
        Matrix of distributions probabilities.
    Returns
    -------
    pcoa : array, shape (`n_dists`, 2)
    """
    dist_matrix = squareform(pdist(distributions, metric=_jensen_shannon))
    return _pcoa(dist_matrix)



def _jensen_shannon(_P, _Q):
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


def _pcoa(pair_dists, n_components=2):
    """Principal Coordinate Analysis,
    aka Classical Multidimensional Scaling
    """
    # code referenced from skbio.stats.ordination.pcoa
    # https://github.com/biocore/scikit-bio/blob/0.5.0/skbio/stats/ordination/_principal_coordinate_analysis.py

    # pairwise distance matrix is assumed symmetric
    pair_dists = np.asarray(pair_dists, np.float64)

    # perform SVD on double centred distance matrix
    n = pair_dists.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = - H.dot(pair_dists ** 2).dot(H) / 2
    eigvals, eigvecs = np.linalg.eig(B)

    # Take first n_components of eigenvalues and eigenvectors
    # sorted in decreasing order
    ix = eigvals.argsort()[::-1][:n_components]
    eigvals = eigvals[ix]
    eigvecs = eigvecs[:, ix]

    # replace any remaining negative eigenvalues and associated eigenvectors with zeroes
    # at least 1 eigenvalue must be zero
    eigvals[np.isclose(eigvals, 0)] = 0
    if np.any(eigvals < 0):
        ix_neg = eigvals < 0
        eigvals[ix_neg] = np.zeros(eigvals[ix_neg].shape)
        eigvecs[:, ix_neg] = np.zeros(eigvecs[:, ix_neg].shape)

    return np.sqrt(eigvals) * eigvecs


 coordinates=js_PCoA(topic_term_dists)











 # Read json file for iterative search
#-------------------------------------------------------

savePath="Y:\\Reference_Databases\\unpaywall\\"
import json_lines
import time
import pandas as pd
# Write and load json file, check time performance
startEntries=int(39e6) # Define from were the processing starts (the next set of processed entries will be saved)
amountofEntries=int(6e7) # How many lines should be processed in total
logEntries=int(1e5) # Intervall of loging
saveEntries=int(1e6) # How many entries are saved per saving iteration
tic1 = time.perf_counter()
tic2 = time.perf_counter()
tic3 = time.perf_counter()
list_of_dfs=[]
df = pd.DataFrame(columns=["doi","year",'is_oa','publisher','journal_name'])
with json_lines.open('Y:\\Reference_Databases\\unpaywall\\unpaywall_snapshot_2021-07-02T151134.jsonl.gz',broken=True) as f:
    for index, item in enumerate(f):
        if (index) < startEntries: # example wait until index 80000 then start processing ((79999+1) < 80000 -> False)
            continue

        else:
            # print(item.keys())
            if (index+1) % logEntries == 0: # example log every 100000 lines (100000 entries = 0:999999 python index)
                toc2 = time.perf_counter()
                print('index = {}'.format(index)," | df appending time in minutes: ",(toc2-tic2)/60, " | Length of list of dfs: ", len(list_of_dfs))
                tic2 = time.perf_counter()
            
            try:
                df = pd.DataFrame(columns=["doi","year",'is_oa','publisher','journal_name'])
                data={"doi":item["doi"], 'genre':item['genre'],"year":item['year'],'title':item['title'],
                            "is_oa":item['is_oa'], "publisher":item['publisher'], "journal_name":item['journal_name']}
                df=df.append(data,ignore_index=True)
                list_of_dfs.append(df)
            except:
                print(index, " could not be processed")

            if (index+1) % saveEntries == 0 and (index+1) != startEntries: # if index reach 79999 it has 80000 entries and is saved as 80000 (100000 entries = 0:999999 python index)
                print("------------------")
                print("Saving to disk")
                df = pd.concat(list_of_dfs, ignore_index=True)
                df.to_pickle(savePath+str(index+1)+".pkl")
                list_of_dfs=[]
                toc3 = time.perf_counter()
                timediff=(toc3-tic3)/60
                print("Processing time in minutes: ", timediff)
                tic3 = time.perf_counter()
                print("------------------")
            
            if (index+1) > amountofEntries: # if index 79999 then it breaks if max amount is 8000
                print("Reached maximum amount of entries")
                break

# toc1=time.perf_counter()
# timediff=(toc1-tic1)/60
# print("Processing time in minutes: ", timediff)

### Processing time: 17.217375168333334 (100'000 entries)
### Processing time:  (1'000'000 entries)


 # Read splitted json file for iterative search
#-------------------------------------------------------

import jsonlines
import time
import pandas as pd
# Write and load json file, check time performance
# startEntries=int(39e6) # Define from were the processing starts (the next set of processed entries will be saved)
# amountofEntries=int(6e7) # How many lines should be processed in total
logEntries=int(1e5) # Intervall of loging
fileName="xaa"
path='Y:\\Reference_Databases\\unpaywall\\'
filepath= path+ fileName
# saveEntries=int(1e6) # How many entries are saved per saving iteration
tic1 = time.perf_counter()
tic2 = time.perf_counter()
tic3 = time.perf_counter()
list_of_dfs=[]
df = pd.DataFrame(columns=["doi","year",'is_oa','publisher','journal_name'])
with jsonlines.open(filepath) as f:
    for index, item in enumerate(f):
            # print(item.keys())
            if (index+1) % logEntries == 0: # example log every 100000 lines (100000 entries = 0:999999 python index)
                toc2 = time.perf_counter()
                print('index = {}'.format(index)," | df appending time in minutes: ",(toc2-tic2)/60, " | Length of list of dfs: ", len(list_of_dfs))
                tic2 = time.perf_counter()
            try:
                df = pd.DataFrame(columns=["doi","year",'is_oa','publisher','journal_name'])
                data={"doi":item["doi"], 'genre':item['genre'],"year":item['year'],'title':item['title'],
                            "is_oa":item['is_oa'], "publisher":item['publisher'], "journal_name":item['journal_name']}
                df=df.append(data,ignore_index=True)
                list_of_dfs.append(df)
            except:
                print(index, " could not be processed")

print("------------------")
print("Saving to disk")
df = pd.concat(list_of_dfs, ignore_index=True)
df.to_pickle(path + fileName +".pkl")
list_of_dfs=[]
toc3 = time.perf_counter()
timediff=(toc3-tic3)/60
print("Processing time in minutes: ", timediff)
tic3 = time.perf_counter()
        
# Read anc concatenate exported pickle files
#-------------------------------------------------------

tic = time.perf_counter()
df=pd.read_pickle(str(int(8e6))+".pkl")
toc = time.perf_counter()
print((toc-tic)/60)
### Processing time: 0.09772834333334686 minutes (1'000'000 entries)


df1=pd.read_pickle(str(int(9e6))+".pkl")
df2=pd.read_pickle(str(int(10e6))+".pkl")
df3=df1.append(df2,ignore_index=True)
len(df3["doi"].unique()) #should be 2e6


df.loc[999999]
len(df["year"].unique())
searchDoi="10.33588/rn.26154.97409"
df["publisher"][df["doi"]==searchDoi]



# Test opeing the gzipped file with other means
#-------------------------------------------------------
import json_lines
with json_lines.open('Y:\\Reference_Databases\\unpaywall\\unpaywall_snapshot_2021-07-02T151134.jsonl.gz',broken=True) as f:
    for index, item in enumerate(f):
        print(index, type(item))
        if index ==3:
            break


import json_lines
with open('Y:\\Reference_Databases\\unpaywall\\unpaywall_snapshot_2021-07-02T151134.jsonl', 'rb') as f:
    for index, item in enumerate(json_lines.reader(f)):
        print(index, type(item))
        if index ==3:
            break



import gzip
import json
with gzip.open('Y:\\Reference_Databases\\unpaywall\\unpaywall_snapshot_2021-07-02T151134.jsonl.gz') as f:
    for index, item in enumerate(f):
        print(index, type(item))
        type(json.loads(item.decode('utf-8')))
        if index ==3:
            break

import jsonlines
with jsonlines.open('Y:\\Reference_Databases\\unpaywall\\unpaywall_snapshot_2021-07-02T151134.jsonl') as f:
    for index, item in enumerate(f):
        print(index, type(item))
        if index ==3:
            break


import jsonlines
with jsonlines.open('Y:\\Reference_Databases\\unpaywall\\xaa') as f:
    for index, item in enumerate(f):
        print(index, type(item))
        if index ==3:
            break

    # for line in f.iter():
    #     print line['doi'] # or whatever else you'd like to do

# Read the splitted gzip files
#-------------------------------------------------------












# corpus to single file and processing with LDA
#-------------------------------------------------------
# Load packages
from Functions.F1_Subsets_and_PreProcessing import Dict_Loader
from Functions.F2_Model_Building import path_creator, log_printer
import pickle
import pandas as pd
import gensim
import time
# import pyLDAvis.gensim_models
import gc
import memory_profiler as mem_profile

from modelConfig_0_49 import *
with open('modelConfig_0_49.py', 'r') as f:
    print(f.read())

# Load the phrases model
bigram=gensim.models.phrases.Phrases.load(path_creator("phraseModel",[Model_Path, StartDir, EndDir, FreezedPhrases_Suffix]))
# Load dictionary
dct=gensim.corpora.Dictionary.load(path_creator("dictionary",[Model_Path, StartDir, EndDir, DictionaryFiltered_Suffix]))

dirNum=0

# Load FilteredMetaData
Meta=pd.read_pickle(path_creator("meta",[IntermediateData_Path, dirNum, MetaDataFiltered_Suffix]))
#Load preprocessed full text
FtPr=Dict_Loader(dirNum, IntermediateData_Path, FtPr_Suffix)
# # Apply meta data filter and delet old dataframe
FtPrFi=[FtPr[DOI] for DOI in list(Meta["DOI"])]

FtPrTest=FtPrFi[0:10000]
doiList=list(Meta["DOI"])[0:10000]
corpus=[dct.doc2bow(doc, allow_update=False) for doc in FtPrTest]

# Initiate empty csv file
import csv

def corpusCSVappender(fileName,corpus):
    # newline='' disables that windows interprests end of lists as new line 
    with open(fileName,'a',newline='') as f:
        writer = csv.writer(f)
        #Append Corpus
        for line in corpus:
            writer.writerow([line])

corpusCSVappender('test2.csv',corpus)

def corpusStream():
    with open("test2.csv", "r") as csv1:
            reader = csv.reader(csv1)
            for row in reader:
                yield row

# corpus2=list(corpusStream())
# type(corpus)
# type(corpus2)
# len(corpus)
# len(corpus2)
# len(corpus[0])
# len(corpus2[0])
# len(corpus[0][0])
# len(corpus2[0][0])

import ast
oneDoc=ast.literal_eval(corpus2[0][0])
ldaModel= gensim.models.ldamodel.LdaModel(id2word=dct, num_topics=nTopics)
ldaModel.update([oneDoc,oneDoc]) # LDA needs multiple docs, current implementation only loads a single document






# Use pandas for appending and loading chunks
#-------------------------------------------------------
# Load packages
from Functions.F1_Subsets_and_PreProcessing import Dict_Loader
from Functions.F2_Model_Building import path_creator, log_printer
import pickle
import pandas as pd
import gensim
import time
# import pyLDAvis.gensim_models
import gc
import memory_profiler as mem_profile
import csv
import ast

from modelConfig_0_49 import *
with open('modelConfig_0_49.py', 'r') as f:
    print(f.read())

# Load the phrases model
bigram=gensim.models.phrases.Phrases.load(path_creator("phraseModel",[Model_Path, StartDir, EndDir, FreezedPhrases_Suffix]))
# Load dictionary
dct=gensim.corpora.Dictionary.load(path_creator("dictionary",[Model_Path, StartDir, EndDir, DictionaryFiltered_Suffix]))

dirNum=0

# Load FilteredMetaData
Meta=pd.read_pickle(path_creator("meta",[IntermediateData_Path, dirNum, MetaDataFiltered_Suffix]))
#Load preprocessed full text
FtPr=Dict_Loader(dirNum, IntermediateData_Path, FtPr_Suffix)
# # Apply meta data filter and delet old dataframe
FtPrFi=[FtPr[DOI] for DOI in list(Meta["DOI"])]

FtPrTest=FtPrFi[0:10000]
doiList=list(Meta["DOI"])[0:10000]
corpus=[dct.doc2bow(doc, allow_update=False) for doc in FtPrTest]


def corpusPandasCSVappender(fileName,corpus,doiList):
    for index,line in enumerate(corpus):
        data = {"DOI":doiList[index], "doc": [line]}
        df = pd.DataFrame(data)
        df.to_csv(fileName, mode='a', index=False, header=False)

def pandasCSVchunkReader(fileName, chunks):
    skiprows=0
    while True:
        chunk = pd.read_csv('testPandas.csv',nrows=chunks,skiprows=skiprows)
        if len(chunk)==0:
            print("Empty chunk -> reader is finished")
            break
        skiprows+=len(chunk)
        yield chunk

def corpusChunkCreator(chunk):
    # Access each item (dataframe) of the chunk and return the list of corpus tuples
    corpusChunk=[ast.literal_eval(chunk.loc[index][1]) for index in range(0,len(chunk))]
    # Access each item (dataframe) of the chunk and return the list of dois
    dois=[chunk.loc[index][0] for index in range(0,len(chunk))]
    return dois,corpusChunk

corpusPandasCSVappender('testPandas.csv',corpus, doiList)

test=pandasCSVchunkReader('testPandas.csv', 1000)

import json
testList=list(test)
len(testList)
len(testList[0])
type(testList[0])
doc=testList[0].iloc[0][1]

res = json.loads(doc)
res = doc.strip('][').split(', ')
res = ast.literal_eval(doc)
res[0].strip(')

import time
tic = time.perf_counter()
ldaModel= gensim.models.ldamodel.LdaModel(id2word=dct, num_topics=nTopics)

for id, chunk in enumerate(test):
    print("Chunk:", id)
    dois,corpusChunk=corpusChunkCreator(chunk)
    print("Length of doi List:", len(dois)," amount of corpus docs: ",len(corpusChunk))
    ldaModel.update(corpusChunk)
toc= time.perf_counter()
print("Time:", toc-tic)
# Processing time: 1492s, 24 min (10'000 docs) -> This is too lon for just 10000 docs



test=pandasCSVchunkReader('testPandas.csv', 1000)

import time
tic = time.perf_counter()
ldaModel=gensim.models.LdaMulticore(id2word=dct, num_topics=nTopics, workers=2)

for id, chunk in enumerate(test):
    print("Chunk:", id)
    dois,corpusChunk=corpusChunkCreator(chunk)
    print("Length of doi List:", len(dois)," amount of corpus docs: ",len(corpusChunk))
    ldaModel.update(corpusChunk)
toc= time.perf_counter()
print("Time:", toc-tic)

# It seems that the multicore version can not be handled within a for loop











# Write list of strings to text file and stream it
#-------------------------------------------------------
# Load packages
from Functions.F1_Subsets_and_PreProcessing import Dict_Loader
from Functions.F2_Model_Building import path_creator, log_printer
import pickle
import pandas as pd
import gensim
import time
# import pyLDAvis.gensim_models
import gc
import memory_profiler as mem_profile
import csv
import ast

from modelConfig_0_49 import *
with open('modelConfig_0_49.py', 'r') as f:
    print(f.read())

# Load the phrases model
bigram=gensim.models.phrases.Phrases.load(path_creator("phraseModel",[Model_Path, StartDir, EndDir, FreezedPhrases_Suffix]))
# Load dictionary
dct=gensim.corpora.Dictionary.load(path_creator("dictionary",[Model_Path, StartDir, EndDir, DictionaryFiltered_Suffix]))

dirNum=0

# Load FilteredMetaData
Meta=pd.read_pickle(path_creator("meta",[IntermediateData_Path, dirNum, MetaDataFiltered_Suffix]))
#Load preprocessed full text
FtPr=Dict_Loader(dirNum, IntermediateData_Path, FtPr_Suffix)
# # Apply meta data filter and delet old dataframe
FtPrFi=[FtPr[DOI] for DOI in list(Meta["DOI"])]

FtPrTest=FtPrFi[0:1000]
# doiList=list(Meta["DOI"])[0:1000]
# corpus=[dct.doc2bow(doc, allow_update=False) for doc in FtPrTest]

outfile= open("test.txt",'a', encoding="utf-8")
# outfile.writelines(FtPrTest[0])
for doc in FtPrTest:
    outfile.writelines("%s " % token for token in doc)
    outfile.write('\n')
outfile.close()


# def textStreamer(filename,dct):
#     for line in open('test.txt', encoding="utf-8"):
#         # assume there's one document per line, tokens separated by whitespace
#         yield dct.doc2bow(line.lower().split())

class MyCorpus:
    def __iter__(self):
        # for line in open('https://radimrehurek.com/mycorpus.txt'):
        for line in open('test.txt', encoding="utf-8"):
            # assume there's one document per line, tokens separated by whitespace
            yield dct.doc2bow(line.lower().split())

class MyCorpus2:
    def __iter__(self):
        for line in FtPrFi:
            # assume there's one document per line, tokens separated by whitespace
            yield dct.doc2bow(line)

# test=textStreamer('mycorpus.txt')
# test=textStreamer('test.txt')
# test=list(test)

ldaModel= gensim.models.ldamodel.LdaModel(id2word=dct, num_topics=nTopics)

import time
tic = time.perf_counter()
docList=[]
# for index, doc in enumerate(textStreamer('mycorpus.txt',dct)):
for index, doc in enumerate(MyCorpus()):
    docList.append(doc)
    if len(docList)==100:
        print("Updating with a chunk of documents at document number:", index)
        ldaModel.update(docList)
        docList=[]
toc= time.perf_counter()
print("Time:", toc-tic)

# Processing time: 574s (1000 docs)


import time
tic = time.perf_counter()
ldaModel.update(MyCorpus(), chunksize=100)
toc= time.perf_counter()
print("Time:", toc-tic)

# Processing time: 355


corpus=MyCorpus()
corpus=MyCorpus2()
import time
tic = time.perf_counter()
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, 
                                      id2word=dct,
                                      num_topics=100,
                                      update_every=1,
                                      chunksize=10000,
                                      passes=1)
toc= time.perf_counter()
print("Time:", toc-tic)

# Processing time: 181 s (1000 docs)
# Processing time: 1841 s (97802 docs) -> 3h / 1M docs