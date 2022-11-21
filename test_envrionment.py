import chunk
import csv
from turtle import shape
from importlib_metadata import files
from itsdangerous import json
import matplotlib
from nbformat import read
import openpyxl
from pyLDAvis import js_PCoA
import unidecode

### os utility
import os

from Functions.F2_Model_Building import MyCorpus
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





#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Multicore processing and paralleization
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

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

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Multicore processing and paralleization 2
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

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



#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Meta Data of preprocessing
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
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


#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Testing of Dictionary and Phrases Model 
# Check if texts can be added to an existing dictinary
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

dirNum=0
# Start Global Timer which times the whole function
tic = time.perf_counter()
FtPr=Dict_Loader(dirNum, IntermediateData_Path, FtPr_Suffix)
#Stop global timer and print
toc = time.perf_counter()
print("Time elapsed in seconds: ", round(toc-tic,4), ", in minutes ", round((toc-tic)/60,4), ", in hours: ", round((toc-tic)/3600,4))

# Start Global Timer which times the whole function
tic = time.perf_counter()
dct=gensim.corpora.Dictionary(list(FtPr.values()))
#Stop global timer and print
toc = time.perf_counter()
print("Time elapsed in seconds: ", round(toc-tic,4), ", in minutes ", round((toc-tic)/60,4), ", in hours: ", round((toc-tic)/3600,4))

print(dct.num_docs,len(dct))
#Out
# 99984


dirNum=1
# Start Global Timer which times the whole function
tic = time.perf_counter()
FtPr=Dict_Loader(dirNum, IntermediateData_Path, FtPr_Suffix)  
dct.add_documents(list(FtPr.values()))
#Stop global timer and print
toc = time.perf_counter()
print("Time elapsed in seconds: ", round(toc-tic,4), ", in minutes ", round((toc-tic)/60,4), ", in hours: ", round((toc-tic)/3600,4))
#Out
# Time elapsed in seconds:  800.4948 , in minutes  13.3416 , in hours:  0.2224

print(dct.num_docs,len(dct))
#Out
# 199976

dirNum=2
# Start Global Timer which times the whole function
tic = time.perf_counter()
FtPr=Dict_Loader(dirNum, IntermediateData_Path, FtPr_Suffix)  
dct.add_documents(list(FtPr.values()))
#Stop global timer and print
toc = time.perf_counter()
print("Time elapsed in seconds: ", round(toc-tic,4), ", in minutes ", round((toc-tic)/60,4), ", in hours: ", round((toc-tic)/3600,4))
#Out
# Time elapsed in seconds:  1164.1192 , in minutes  19.402 , in hours:  0.3234

print(dct.num_docs,len(dct))
#Out
# 299975 2269904



#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Testing of Dictionary and Phrases Model 
# Test if it is faster to merge dictionaries instead of adding documents
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

dirNum=0
# Start Global Timer which times the whole function
tic = time.perf_counter()
FtPr=Dict_Loader(dirNum, IntermediateData_Path, FtPr_Suffix) 
dct=gensim.corpora.Dictionary(list(FtPr.values()))
dirNum=1
FtPr=Dict_Loader(dirNum, IntermediateData_Path, FtPr_Suffix)  
dct.add_documents(list(FtPr.values()))
print(dct.num_docs,len(dct))
#Stop global timer and print
toc = time.perf_counter()
print("Time elapsed in seconds: ", round(toc-tic,4), ", in minutes ", round((toc-tic)/60,4), ", in hours: ", round((toc-tic)/3600,4))

#Out
# 199976 2185638
# Time elapsed in seconds:  945.4743 , in minutes  15.7579 , in hours:  0.2626

dirNum=0
# Start Global Timer which times the whole function
tic = time.perf_counter()
FtPr=Dict_Loader(dirNum, IntermediateData_Path, FtPr_Suffix) 
dct1=gensim.corpora.Dictionary(list(FtPr.values()))
dirNum=1
# Start Global Timer which times the whole function
tic = time.perf_counter()
FtPr=Dict_Loader(dirNum, IntermediateData_Path, FtPr_Suffix) 
dct2=gensim.corpora.Dictionary(list(FtPr.values()))
dct1.merge_with(dct2)
print(dct1.num_docs,len(dct1))
#Stop global timer and print
toc = time.perf_counter()
print("Time elapsed in seconds: ", round(toc-tic,4), ", in minutes ", round((toc-tic)/60,4), ", in hours: ", round((toc-tic)/3600,4))

#Out
# 199976 3426641
# Time elapsed in seconds:  1174.3321 , in minutes  19.5722 , in hours:  0.3262

'''
Notes:
It seems that the output of mergin is a transformer not a dict
'''



#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Testing of Dictionary and Phrases Model 
# Test if bulding a corpus is faster with a smaller dictionary
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

dirNum=0
# Start Global Timer which times the whole function
tic = time.perf_counter()
FtPr=Dict_Loader(dirNum, IntermediateData_Path, FtPr_Suffix)  
corpus = [dct.doc2bow(text) for text in list(FtPr.values())]
#Stop global timer and print
toc = time.perf_counter()
print("Time elapsed in seconds: ", round(toc-tic,4), ", in minutes ", round((toc-tic)/60,4), ", in hours: ", round((toc-tic)/3600,4))
#Out
# Time elapsed in seconds:  1643.5739 , in minutes  27.3929 , in hours:  0.4565

print(dct.num_docs,len(dct))
#Out
# 299975 2269904

dct.filter_extremes(keep_n=500000)
print(dct.num_docs,len(dct))
#Out
# 299975 500000

# Start Global Timer which times the whole function
tic = time.perf_counter()
corpus2 = [dct.doc2bow(text) for text in list(FtPr.values())]
#Stop global timer and print
toc = time.perf_counter()
print("Time elapsed in seconds: ", round(toc-tic,4), ", in minutes ", round((toc-tic)/60,4), ", in hours: ", round((toc-tic)/3600,4))
#Out
# Time elapsed in seconds:  830.7225 , in minutes  13.8454 , in hours:  0.2308



#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Testing of Dictionary and Phrases Model 
# Test phrases model with freezing
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

dirNum=1
# Start Global Timer which times the whole function
tic = time.perf_counter()
FtPr=Dict_Loader(dirNum, IntermediateData_Path, FtPr_Suffix)  
# Define bigram and Trigram models
bigram = gensim.models.phrases.Phrases(list(FtPr.values()), min_count=3)
# trigram = gensim.models.phrases.Phrases(bigram[list(FtPr.values())], min_count=3)
# tetragram = gensim.models.phrases.Phrases(trigram[list(FtPr.values())], min_count=3)
# Apply the bigram and trigram model to each preprocessed tokenized list
# FtPrNg=[tetragram[trigram[bigram[doc]]] for doc in list(FtPr.values())]
FtPrNg=[bigram[doc] for doc in list(FtPr.values())]
#Stop global timer and print
toc = time.perf_counter()
print("Time elapsed in seconds: ", round(toc-tic,4), ", in minutes ", round((toc-tic)/60,4), ", in hours: ", round((toc-tic)/3600,4))
#Out
# Time elapsed in seconds:  2275.3597 , in minutes  37.9227 , in hours:  0.632

print(list(FtPr.values())[0][0:100])
#Out
# ['li', 'chien', 'chen', 'et', 'al', 'effect', 'heat', 'treatment', 'ni', 'au', 'ohmic', 'contact', 'phys', 'stat', 'sol', 'subject', 'classification', 'cg', 'effect', 'heat', 'treatment', 'ni', 'au', 'ohmic', 'contact', 'type', 'gan', 'li', 'chien', 'chen', 'jin', 'kuo', 'ho', 'fu', 'rong', 'chen', 'ji', 'jung', 'kai', 'li', 'chang', 'chang', 'shyang', 'jong', 'chien', 'chiu', 'chao', 'nien', 'huang', 'kwang', 'kuo', 'shih', 'department', 'engineering', 'system', 'science', 'national', 'tsing', 'hua', 'university', 'hsinchu', 'taiwan', 'department', 'material', 'science', 'engineering', 'national', 'chiao', 'tung', 'university', 'hsinchu', 'taiwan', 'opto', 'electronics', 'system', 'laboratory', 'industrial', 'technology', 'research', 'institute', 'chutung', 'hsinchu', 'taiwan', 'received', 'july', 'effect', 'heat', 'treatment', 'temperature', 'microstructure', 'specific', 'contact', 'resistance', 'oxidized', 'ni', 'nm', 'au', 'nm', 'contact', 'type']

print(FtPrNg[0][0:100])
#Out
# ['li', 'chien', 'chen', 'et_al', 'effect', 'heat', 'treatment', 'ni', 'au', 'ohmic_contact', 'phys_stat', 'sol_subject', 'classification', 'cg', 'effect', 'heat', 'treatment', 'ni', 'au', 'ohmic_contact', 'type', 'gan', 'li', 'chien', 'chen_jin', 'kuo', 'ho', 'fu', 'rong_chen', 'ji', 'jung', 'kai', 'li', 'chang_chang', 'shyang', 'jong', 'chien', 'chiu', 'chao', 'nien', 'huang', 'kwang', 'kuo', 'shih', 'department', 'engineering', 'system', 'science', 'national_tsing', 'hua_university', 'hsinchu_taiwan', 'department', 'material', 'science_engineering', 'national_chiao', 'tung_university', 'hsinchu_taiwan', 'opto_electronics', 'system', 'laboratory', 'industrial', 'technology', 'research', 'institute', 'chutung', 'hsinchu_taiwan', 'received_july', 'effect', 'heat', 'treatment', 'temperature', 'microstructure', 'specific', 'contact', 'resistance', 'oxidized', 'ni', 'nm', 'au', 'nm', 'contact', 'type', 'gan', 'wa', 'investigated', 'minimum', 'specific', 'contact', 'resistance', 'rc', 'obtained', 'wa', 'cm', 'heat', 'treating', 'air', 'min', 'cross_sectional', 'microstructure', 'heat']

'''
Notes:
with freeze
'''

dirNum=1
# Start Global Timer which times the whole function
tic = time.perf_counter()
FtPr=Dict_Loader(dirNum, IntermediateData_Path, FtPr_Suffix)  
# Define bigram and Trigram models
bigram = gensim.models.phrases.Phrases(list(FtPr.values()), min_count=3)
# trigram = gensim.models.phrases.Phrases(bigram[list(FtPr.values())], min_count=3)
# tetragram = gensim.models.phrases.Phrases(trigram[list(FtPr.values())], min_count=3)
# Freeze Model
frozen_bigram_model = bigram.freeze()
# frozen_trigram_model = trigram.freeze()
# frozen_tetragram_model = tetragram.freeze()
# Apply the bigram and trigram model to each preprocessed tokenized list
# FtToPrNg=[frozen_tetragram_model[frozen_trigram_model[frozen_bigram_model[doc]]] for doc in list(FtPr.values())]
FtToPrNg=[frozen_bigram_model[doc] for doc in list(FtPr.values())]
#Stop global timer and print
toc = time.perf_counter()
print("Time elapsed in seconds: ", round(toc-tic,4), ", in minutes ", round((toc-tic)/60,4), ", in hours: ", round((toc-tic)/3600,4))
#Out
# Time elapsed in seconds:  686.8023 , in minutes  11.4467 , in hours:  0.1908



#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Testing of Dictionary and Phrases Model 
# Test adding new dataset to phrase model
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

# Start Global Timer which times the whole function
tic = time.perf_counter()
dirNum=0
FtPr=Dict_Loader(dirNum, IntermediateData_Path, FtPr_Suffix)
FtPr=list(FtPr.values())[0:20000]
# Define bigram and Trigram models
bigram = gensim.models.phrases.Phrases(FtPr, min_count=3)
print(len(bigram.vocab))
dirNum=1
FtPr1=Dict_Loader(dirNum, IntermediateData_Path, FtPr_Suffix)  
FtPr1=list(FtPr1.values())[0:20000]
# Define bigram and Trigram models
bigram.add_vocab(FtPr1)
print(len(bigram.vocab))
# Apply the bigram and trigram model to each preprocessed tokenized list
# FtPrNg=[tetragram[trigram[bigram[doc]]] for doc in list(FtPr.values())]
FtPrNg=[bigram[doc] for doc in FtPr]
FtPrNg1=[bigram[doc] for doc in FtPr1]
#Stop global timer and print
toc = time.perf_counter()
print("Time elapsed in seconds: ", round(toc-tic,4), ", in minutes ", round((toc-tic)/60,4), ", in hours: ", round((toc-tic)/3600,4))
#Out
# 17424778
# 28701000
# Time elapsed in seconds:  565.2361 , in minutes  9.4206 , in hours:  0.157

print(FtPrNg[50][0:100])
#Out
# ['international_journal', 'geriatric_psychiatry', 'vol', 'schizophrenia', 'onset', 'extreme', 'adult', 'life', 'david', 'castle', 'simon', 'wessely', 'robert', 'howard', 'robin', 'murray', 'department', 'psychological_medicine', 'king_college', 'hospital', 'institute_psychiatry', 'london_uk', 'section', 'old_age', 'psychiatry', 'institute_psychiatry', 'london_uk', 'abstract_objective', 'de_ne', 'epidemiology', 'phenomenology', 'premorbid', 'risk_factor', 'patient', 'rst', 'manifestation', 'schizophrenia', 'like', 'illness', 'age_year', 'compare', 'patient', 'onset', 'age_year', 'design', 'setting', 'subject', 'contact', 'non', 'ective_psychotic', 'illness', 'across', 'age_onset', 'ascertained', 'psychiatric', 'case', 'register', 'patient', 'rediagnosed_according', 'operationalized_criterion', 'psychotic_illness', 'early_late', 'onset', 'compared', 'main', 'outcome_measure', 'phenomenological', 'premorbid', 'aetiological', 'parameter', 'compared', 'two', 'group', 'using', 'risk', 'ratio', 'con_dence', 'interval', 'result', 'late_onset', 'patient', 'compared', 'early_onset', 'counterpart', 'likely', 'female', 'good_premorbid', 'functioning', 'developmental', 'history', 'exhibit', 'persecutory_delusion', 'hallucination', 'le_likely', 'negative', 'schizophrenic_symptom', 'positive', 'family_history', 'schizophrenia', 'su_ered']

# Start Global Timer which times the whole function
tic = time.perf_counter()
dirNum=0
FtPr=Dict_Loader(dirNum, IntermediateData_Path, FtPr_Suffix)
FtPr=list(FtPr.values())[0:20000]
# Define bigram and Trigram models
bigram = gensim.models.phrases.Phrases(FtPr, min_count=3)
print(len(bigram.vocab))
dirNum=1
FtPr1=Dict_Loader(dirNum, IntermediateData_Path, FtPr_Suffix)  
FtPr1=list(FtPr1.values())[0:20000]
# Define bigram and Trigram models
bigram.add_vocab(FtPr1)
print(len(bigram.vocab))
# Freeze Model
frozen_bigram_model = bigram.freeze()
# Apply the bigram and trigram model to each preprocessed tokenized list
# FtPrNg=[tetragram[trigram[bigram[doc]]] for doc in list(FtPr.values())]
FtPrNg=[frozen_bigram_model[doc] for doc in FtPr]
FtPrNg1=[frozen_bigram_model[doc] for doc in FtPr1]
#Stop global timer and print
toc = time.perf_counter()
print("Time elapsed in seconds: ", round(toc-tic,4), ", in minutes ", round((toc-tic)/60,4), ", in hours: ", round((toc-tic)/3600,4))
#Out
# 17424778
# 28701000
# Time elapsed in seconds:  760.2073 , in minutes  12.6701 , in hours:  0.2112

print(FtPrNg[50][0:100])
#Out
# ['international_journal', 'geriatric_psychiatry', 'vol', 'schizophrenia', 'onset', 'extreme', 'adult', 'life', 'david', 'castle', 'simon', 'wessely', 'robert', 'howard', 'robin', 'murray', 'department', 'psychological_medicine', 'king_college', 'hospital', 'institute_psychiatry', 'london_uk', 'section', 'old_age', 'psychiatry', 'institute_psychiatry', 'london_uk', 'abstract_objective', 'de_ne', 'epidemiology', 'phenomenology', 'premorbid', 'risk_factor', 'patient', 'rst', 'manifestation', 'schizophrenia', 'like', 'illness', 'age_year', 'compare', 'patient', 'onset', 'age_year', 'design', 'setting', 'subject', 'contact', 'non', 'ective_psychotic', 'illness', 'across', 'age_onset', 'ascertained', 'psychiatric', 'case', 'register', 'patient', 'rediagnosed_according', 'operationalized_criterion', 'psychotic_illness', 'early_late', 'onset', 'compared', 'main', 'outcome_measure', 'phenomenological', 'premorbid', 'aetiological', 'parameter', 'compared', 'two', 'group', 'using', 'risk', 'ratio', 'con_dence', 'interval', 'result', 'late_onset', 'patient', 'compared', 'early_onset', 'counterpart', 'likely', 'female', 'good_premorbid', 'functioning', 'developmental', 'history', 'exhibit', 'persecutory_delusion', 'hallucination', 'le_likely', 'negative', 'schizophrenic_symptom', 'positive', 'family_history', 'schizophrenia', 'su_ered']





#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Test log functions
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
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







#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# If conda is not activated, invoke python then hit ctrl+Z then type conda activate to activate conda
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------






#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Try applying generators for data streaming
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
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





#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# js_PCoA
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
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










#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Read json file for iterative search
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

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





#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
 # Read splitted json file for iterative search
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

import jsonlines
import time
import pandas as pd
import os
import regex as re




# Write and load json file, check time performance
# startEntries=int(39e6) # Define from were the processing starts (the next set of processed entries will be saved)
# amountofEntries=int(6e7) # How many lines should be processed in total
logEntries=int(1e5) # Intervall of loging
# fileName="xaa"
path='Y:\\Reference_Databases\\unpaywall\\'
# saveEntries=int(1e6) # How many entries are saved per saving iteration
fileNameList=os.listdir(path)
for name in fileNameList:
    if ".pkl" in name or ".jsonl" in name or ".gz" in name or ".json.gz" in name:
        fileNameList.remove(name)
        print(name, "is removed")

for fileName in fileNameList:
    filepath= path+ fileName
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




#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------  
# Read and concatenate exported pickle files
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

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




#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Test opeing the gzipped file with other means
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
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








#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# corpus to single file and processing with LDA
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
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





#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Use pandas for appending and loading chunks
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
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









#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Write list of strings to text file and stream it
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
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








#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# ## Test: Search and Match the preprocessing meta data and reference database meta data
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

import pandas as pd
import pickle
import time
import os

'''
Notes:
Data frame with 1M lines load in approx 10s
Goal is it to assign the filepath to the reference database meta data which then can be sampled
The DOI from the data side are filtered based on the preprocessed filter meta data file
Concatenation of 10 reference database files is 40s

'''

# Get a list off all the unpaywall metadata .pkl files
path='Y:\\Reference_Databases\\unpaywall\\'
fileNameList=os.listdir(path)
for name in fileNameList:
    if ".pkl" not in name:
        fileNameList.remove(name)
        print(name, "is removed")

# Concatenate the firs 10 metadata files
tic = time.perf_counter()
df3=pd.concat([pd.read_pickle(path+name) for name in fileNameList[0:10]], ignore_index=True)
toc = time.perf_counter()
print(toc-tic)

# Load one of the Filtered Meta Data Files and the doi file path dictionary
tic = time.perf_counter()
path='Y:\\IntermediateData\\040_MetaDataFiltered.pkl'
toc = time.perf_counter()
print(toc-tic)
df2=pd.read_pickle(path)
path='Y:\\IntermediateData\\040_DOI_Path_Dict.pkl'
tic = time.perf_counter()
with open(path, 'rb') as handle:
    interDict = pickle.load(handle)
toc = time.perf_counter()
print(toc-tic)

# Get a reduced doi pathy dicitonary based on the preprocessed filtered meta data
keyList=list(df2["DOI"])
valueList=[interDict[key] for key in keyList]
len(keyList)
len(valueList)
filteredDct = dict(zip(keyList, valueList))


# Create a new empyt columen in the df3 dataframe which will hold the file paths
df3["FilePath"]=None

# Create pandas series containing the found file paths
filepathSeries=df3['doi'].map(filteredDct)
filepathSeries.value_counts()

# Map the non-NaN values from the filepathSeries to the df3 dataframe. But only map there is not already a file path assigned
# Map the file path series to the file path values in the df3 dataframe, but do not map NaN to None
df3["FilePath"]=filepathSeries.map(lambda x: x if pd.notnull(x) else None)
df3["FilePath"].value_counts()


# Test with an additional set

# Create a new empyt columen in the df3 dataframe which will hold the file paths
df3["FilePath"]=None

# Load one of the Filtered Meta Data Files and the doi file path dictionary
tic = time.perf_counter()
path='Y:\\IntermediateData\\041_MetaDataFiltered.pkl'
toc = time.perf_counter()
print(toc-tic)
df4=pd.read_pickle(path)
path='Y:\\IntermediateData\\041_DOI_Path_Dict.pkl'
tic = time.perf_counter()
with open(path, 'rb') as handle:
    interDict2 = pickle.load(handle)
toc = time.perf_counter()
print(toc-tic)

# Get a reduced doi pathy dicitonary based on the preprocessed filtered meta data
keyList=list(df4["DOI"])
valueList=[interDict2[key] for key in keyList]
len(keyList)
len(valueList)
filteredDct2 = dict(zip(keyList, valueList))

# Concatenate filteredDct and filteredDct2
len(filteredDct)
len(filteredDct2)
filteredDct.update(filteredDct2)
len(filteredDct)

# Create pandas series containing the found file paths
filepathSeries=df3['doi'].map(filteredDct)
filepathSeries.value_counts()

# Map the non-NaN values from the filepathSeries to the df3 dataframe. But only map there is not already a file path assigned
# Map the file path series to the file path values in the df3 dataframe, but do not map NaN to None
df3["FilePath"]=filepathSeries.map(lambda x: x if pd.notnull(x) else None)
df3["FilePath"].value_counts()

# Check the sum which indicates no dublicates
df3["FilePath"].value_counts().sum()

# Check validity of the doi Matching
df3.FilePath.first_valid_index()
df3.loc[1221]
df3["doi"].loc[1221]
filePath=df3["FilePath"].loc[1221]

#Print title of the article
df3["title"].loc[1221]

# Print corresponding data and check if it matches
with open(filePath, "r", encoding="utf8") as f:
    contents = f.read()
    len(contents)
    print(contents[0:100])

# -> valid, but the file Path is differently formated !!!!!!!!!




#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# ## Concatenate 000-099 meta data files and doi path dictionaries together
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
import pandas as pd
import pickle
import time
import os

# Define input paths and file names
IntermediateData_Path="Y:\\IntermediateData\\"
doiPath_Suffix="_DOI_Path_Dict.pkl"
doiPathFiltered_Suffix="_DOI_Path_DictFiltered.pkl"
metaData_Suffix="_MetaDataFiltered.pkl"

# Set the dirs to save doi and paths (Dictionaries have to be generated to the corresponding directories beforehand)
StartDir=0
EndDir=99

### Concatendated doi path dict

# Define Path for saving new concatenated dictionary
saveNameDict=(IntermediateData_Path + str(StartDir).zfill(3) + "_" + str(EndDir).zfill(3) + doiPath_Suffix).replace(" ","")

# create a binary pickle file 
f = open(saveNameDict,"wb")
# Init new dict
doiPathDict={}

# Iterate trough data directories
for dirNum in range(StartDir,EndDir+1):
    # Bring for example 27 into the form of "027"
    dirNum=str(dirNum).zfill(3)
    # Create path to dictionary
    openName=(IntermediateData_Path + dirNum + doiPath_Suffix).replace(" ","")
    # Open the dictionary
    with open(openName, 'rb') as handle:
        interDict = pickle.load(handle)
    # Append the dictionary
    doiPathDict = {**doiPathDict, **interDict}
    print(len(interDict), len(doiPathDict))

# write the python object (dict) to pickle file
pickle.dump(doiPathDict,f)
# close file
f.close()


### Read multple filtered preprocessed meta data files and concatenate them into one dataframe

# Define Path for saving new concatenated dataframe
saveNameDf=(IntermediateData_Path + str(StartDir).zfill(3) + "_" + str(EndDir).zfill(3) + metaData_Suffix).replace(" ","")

# Create a list with the file names
fileList=[]
for dirNum in range(StartDir,EndDir+1):
    # Bring for example 27 into the form of "027"
    dirNum=str(dirNum).zfill(3)
    # Create path to dictionary
    openName=(IntermediateData_Path + dirNum + metaData_Suffix).replace(" ","")
    # Append the file name to the list
    fileList.append(openName)
    print(len(fileList))

# Concatenate all metadata files
tic = time.perf_counter()
df=pd.concat([pd.read_pickle(name) for name in fileList], ignore_index=True)
toc = time.perf_counter()
print(toc-tic)

# Filter the new doi path dict based on the new concatenated dataframe
    
# Define Path for saving new filtered dictionary
loadNameDict=(IntermediateData_Path + str(StartDir).zfill(3) + "_" + str(EndDir).zfill(3) + doiPath_Suffix).replace(" ","")
with open(loadNameDict, 'rb') as handle:
    interDict = pickle.load(handle)


# Get a reduced doi pathy dicitonary based on the preprocessed filtered meta data
keyList=list(df["DOI"])
valueList=[interDict[key] for key in keyList]
len(keyList)
len(valueList)
filteredDct = dict(zip(keyList, valueList))

saveNameDict=(IntermediateData_Path + str(StartDir).zfill(3) + "_" + str(EndDir).zfill(3) + doiPathFiltered_Suffix).replace(" ","")

# create a binary pickle file 
f = open(saveNameDict,"wb")
# write the python object (dict) to pickle file
pickle.dump(filteredDct,f)
# close file
f.close()

# -> This continuous in the next section

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# ## Concatenate 0-50 of the reference database meta data files
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

import pandas as pd
import pickle
import time
import os

'''
Notes:
It seems it is not possible to concatenate all of the files dues to the memory limit.
Concatenation of 20 reference database files is 52s
Concatenation of 50 reference database files is 228s
'''

path='Y:\\Reference_Databases\\unpaywall\\'

### Concatendated reference database metadata file

# Get a list off all the unpaywall metadata .pkl files
fileNameList=os.listdir(path)
for name in fileNameList:
    if ".pkl" not in name:
        fileNameList.remove(name)
        print(name, "is removed")

# Concatenate all metadata files
tic = time.perf_counter()
df3=pd.concat([pd.read_pickle(path+name) for name in fileNameList[0:50]], ignore_index=True)
toc = time.perf_counter()
print(toc-tic)

# -> This continuous in the next section


#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# ## Match all of the reference database dois to a doi path
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

import pandas as pd
import pickle
import time
import os

# Define input paths and file names
doiPathFiltered_Suffix="_DOI_Path_DictFiltered.pkl"
path='Y:\\Reference_Databases\\unpaywall\\'
IntermediateData_Path="Y:\\IntermediateData\\"

# Set the dirs to save doi and paths (Dictionaries have to be generated to the corresponding directories beforehand)
StartDir=0
EndDir=99

# Create a new empyt columen in the df3 dataframe which will hold the file paths
df3["FilePath"]=None

# Define Path for saving new filtered dictionary
loadNameDict=(IntermediateData_Path + str(StartDir).zfill(3) + "_" + str(EndDir).zfill(3) + doiPathFiltered_Suffix).replace(" ","")
with open(loadNameDict, 'rb') as handle:
    filteredDct = pickle.load(handle)
len(filteredDct)

# Create pandas series containing the found file paths


# Map the non-NaN values from the filepathSeries to the df3 dataframe. But only map there is not already a file path assigned
# Map the file path series to the file path values in the df3 dataframe, but do not map NaN to None
df3["FilePath"]=filepathSeries.map(lambda x: x if pd.notnull(x) else None)
df3["FilePath"].value_counts()

# Check the sum which indicates no dublicates
df3["FilePath"].value_counts().sum()

saveName='Y:\\Reference_Databases\\unpaywall\\' + fileNameList[0] + "-" + fileNameList[50]+ "_"+ str(StartDir).zfill(3) + "-" + str(EndDir).zfill(3) + ".pkl"

# Save the dataframe to a pickle file
df3.to_pickle(saveName)

# -> This continuous in the next section

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# ## Sample multiple matched dois from the reference database and check if the path is correct
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------


import pandas as pd
import pickle
import time
import os

# Define input paths and file names
# doiPathFiltered_Suffix="_DOI_Path_DictFiltered.pkl"
# path='Y:\\Reference_Databases\\unpaywall\\'
# IntermediateData_Path="Y:\\IntermediateData\\"

# Set the dirs to save doi and paths (Dictionaries have to be generated to the corresponding directories beforehand)
StartDir=0
EndDir=99

loadName='Y:\\Reference_Databases\\unpaywall\\' + fileNameList[0] + "-" + fileNameList[50]+ "_"+ str(StartDir).zfill(3) + "-" + str(EndDir).zfill(3) + ".pkl"

tic = time.perf_counter()
df=pd.read_pickle(loadName)
toc = time.perf_counter()
print(toc-tic) #192s

# Create a subset of the dataframe by sampling randomly n rows which have non-None file paths
df2=df[df["FilePath"].notnull()]
df3=df2.sample(n=100)
df3.head()

# Realized that I should also have linked the FtPr data !!!!!!!





#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Test: if it is better to try find a match in the reference databased based on the 
# Preprocessed metadata instead the other way around
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

# Check if all of the reference database files can be put into a single file
import pandas as pd
import pickle
import time
import os

'''
Notes:
Concatenation of 20 reference database files is 52s
Concatenation of 50 reference database files is 228s
Concatenation of 100 reference database files is 466s
Concatenation of all of the meta data 1040s / 17min -> But can not be saved due to memory error!

'''
path='Y:\\Reference_Databases\\unpaywall\\'

### Concatendated reference database metadata file
# Get a list off all the unpaywall metadata .pkl files
fileNameList=os.listdir(path)
for name in fileNameList:
    if ".pkl" not in name:
        fileNameList.remove(name)
        print(name, "is removed")
fileNameList.remove("xaa.pkl-xby.pkl_000-099.pkl")
fileNameList.remove("xaa.pkl-xew.pkl.pkl")

# Concatenate all metadata files
tic = time.perf_counter()
df=pd.concat([pd.read_pickle(path+name) for name in fileNameList], ignore_index=True)
toc = time.perf_counter()
print(toc-tic)

# Save the dataframe 
saveName='Y:\\Reference_Databases\\unpaywall\\' + fileNameList[0] + "-" + fileNameList[-1] + ".pkl"

# Save the dataframe to a pickle file
df.to_pickle(saveName)

# Load it again and check how much time passes
tic = time.perf_counter()
df=pd.read_pickle(saveName)
toc = time.perf_counter()
print(toc-tic) # take 1142s or 17min

# Load one of the preprocessed meta data files
path='Y:\\IntermediateData\\040_MetaDataFiltered.pkl'
df2=pd.read_pickle(path)

# For every value in the "DOI" column of the df2 dataframe, extract the corresponding row from the df dataframe
df3=df2.merge(df, how='left', left_on='DOI', right_on='doi')
df3.head()

# For every string the df2 doi column extract the row in the df dataframe which has the same doi
df3=df2.merge(df, how='left', left_on='doi', right_on='doi')
df3.head()

# check if the "title" column only has unique values
df3["title"].nunique()

# check if "doi" column only has unique values
df3["doi"].nunique()

# Get an example of title which comes up more than once
df3["title"].value_counts().head(100)

# Get a list of non unique values of the "title" column of the df3 dataframe
nonUniTitleList=list(df3["title"].value_counts().index[df3["title"].value_counts()>1])
len(nonUniTitleList)
nonUniTitleList[-1]
df3[df3["title"]=="Transformations of sugars in alkaline solutions"]

# Drop every row if the corresponding title value is in the nonUniTitleList
df4=df3[~df3["title"].isin(nonUniTitleList)]

# Check if the new dataframe now has unique values in the "title" column
df4["title"].nunique()







#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Apply dask for the reference database meta data concatenation
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

import dask.dataframe as dd
import pandas as pd
import os
import time
import pickle
from Functions.F3_Reference_Databases_and_Alignment import CleanJoinedDf

from modelConfig_0_99 import *

path='Y:\\Reference_Databases\\unpaywall\\splitted_old'
csvPath='Y:\\Reference_Databases\\unpaywall\\xaa-xba.csv'
MetaDataJoined_Suffix="_MetaDataJoined.pkl"
LengthInfoDf_Suffix="_LengthInfoDf.pkl"
LengthInfoDf_Suffix2="_LengthInfoDf2.csv"


# Get a list off all the unpaywall metadata .pkl files
fileNameList=os.listdir(path)
for name in fileNameList:
    if ".pkl" not in name:
        fileNameList.remove(name)
        print(name, "is removed")


# Add the first file with headers to the csv file
df=pd.read_pickle(path +"\\"+ fileNameList[0])
df.to_csv(csvPath, mode='a', index=False, header=True)

# Append all file in the fileNameList into a single csv file
for name in fileNameList[1::]:
    df=pd.read_pickle(path+"\\"+name)
    df.to_csv(csvPath, mode='a', index=False, header=False)
    print(name, "is appended")



# Read the created csv file with dask
RefDf=dd.read_csv(csvPath, assume_missing=True) #some year dates are missing !!!

# show the amount of dask partitions
RefDf.npartitions

# Compute head
RefDf.head(n=5, npartitions=1, compute=True)

# Get length of the RefDf dataframe with dask
RefDf.shape[0].compute()


# Iterate trough every directory and load the corresponding metadta file
StartDir=0
EndDir=9

# Iterate trough data directories
for dirNum in range(StartDir,EndDir+1):

    # Create path to dictionary and read the preprocessed metadata file
    path=(IntermediateData_Path + str(dirNum).zfill(3) + MetaDataFiltered_Suffix)
    PpMetaDf=pd.read_pickle(path)

    #Harmonize Column Naming
    PpMetaDf.columns=["doi", "TokenAmount", "Language"]

    # Create inner join of the PpMetaDf and the RefDf
    join = RefDf.merge(PpMetaDf, how="inner", on=["doi"])
    tic = time.perf_counter()
    JoinedDf=join.compute()
    toc = time.perf_counter()
    print(toc-tic) 

    # # Get length of the joined dataframe
    # JoinedDfLength=len(JoinedDf)
    
    # # Geth the amount of non unique value in the "title" column
    # nonUniTitleList=list(JoinedDf["title"].value_counts().index[JoinedDf["title"].value_counts()>1])
    # NonUniTitleAmount=len(nonUniTitleList)
    # # Get the amount of unique values in the "doi" column
    # nonUniDoiList=list(JoinedDf["doi"].value_counts().index[JoinedDf["doi"].value_counts()>1])
    # NonUniDOIAmount=len(nonUniDoiList)
    # # Remove all of the rows in the JoinedDf with titles which are not unique 
    # JoinedDf_Cleaned_1=JoinedDf[~JoinedDf["title"].isin(nonUniTitleList)]
    # # Count the amount of rows in the cleaned dataframe
    # JoinedDfLength_afterTitleCleaning=len(JoinedDf_Cleaned_1)

    # # Check if after removing non unique titles if there are still non unique dois
    # nonUniDoiList=list(JoinedDf_Cleaned_1["doi"].value_counts().index[JoinedDf_Cleaned_1["doi"].value_counts()>1])
    # NonUniDOIAmount_afterTitleCleaning=len(nonUniDoiList)
    # # Remove all of the rows in the JoinedDf with dois which are not unique 
    # JoinedDf_Cleaned_2=JoinedDf_Cleaned_1[~JoinedDf_Cleaned_1["doi"].isin(nonUniDoiList)]
    # # Count the amount of rows in the cleaned dataframe
    # JoinedDfLength_afterDOICleaning=len(JoinedDf_Cleaned_2)

    # # Get the amount on NaN values in the "title" column
    # NaNasTitleList=list(JoinedDf_Cleaned_2["doi"][JoinedDf_Cleaned_2["title"].isna()==True])
    # NaNasTitleAmount=len(NaNasTitleList)
    # # Remove all of the rows in the JoinedDf with titles which are NaN
    # JoinedDf_Cleaned_3=JoinedDf_Cleaned_2[~JoinedDf_Cleaned_2["doi"].isin(NaNasTitleList)]
    # # Count the amount of rows in the cleaned dataframe
    # JoinedDfLength_afterNaNTitleCleaning=len(JoinedDf_Cleaned_3)


    # # Get the amount of NaN vlaues in the "year" column of the joined dataframe
    # NaNasYearList=list(JoinedDf_Cleaned_3["doi"][JoinedDf_Cleaned_3["year"].isna()==True])
    # NaNasYearAmount=len(NaNasYearList)
    # # Remove all of the rows in the JoinedDf with years which are NaN
    # JoinedDf_Cleaned_4=JoinedDf_Cleaned_3[~JoinedDf_Cleaned_3["doi"].isin(NaNasYearList)]
    # # Count the amount of rows in the cleaned dataframe
    # JoinedDfLength_afterNaNYearCleaning=len(JoinedDf_Cleaned_4)

    # # Turn the year values in the "year" column into integers
    # JoinedDf_Cleaned_4["year"]=JoinedDf_Cleaned_4["year"].astype(int)

    # # Creat path to the _DOI_Path_Dict.pkl file based on the current dirNum and then load it
    # path=(IntermediateData_Path + str(dirNum).zfill(3) + DOIPath_Suffix)
    # DOI_Path_Dict=pickle.load(open(path, "rb"))
    # # Create a new column in the JoinedDf_Cleaned_4 dataframe which contains the paths to the raw textfiles based on the doi
    # JoinedDf_Cleaned_4["text_path"]=JoinedDf_Cleaned_4["doi"].map(DOI_Path_Dict)

    # # Rename the "TokenAmount" column to "token_amount" and the "Language" column to "language"
    # JoinedDf_Cleaned_4.rename(columns={"TokenAmount": "token_amount", "Language": "language"}, inplace=True)

    # # Check each column for the amount of missing values
    # MissingValues=JoinedDf_Cleaned_4.isna().sum().sum()

    # JoinedDfLength # Length of the original joined dataframe
    # NonUniTitleAmount # Amount of non unique titles
    # NonUniDOIAmount # Amount of non unique dois
    # JoinedDfLength_afterTitleCleaning # Length of the dataframe after removing non unique titles
    # NonUniDOIAmount_afterTitleCleaning # Amount of non unique dois after removing non unique titles
    # JoinedDfLength_afterDOICleaning # Length of the dataframe after removing non unique dois
    # NaNasTitleAmount # Amount of NaN values in the title column
    # JoinedDfLength_afterNaNTitleCleaning # Length of the dataframe after removing NaN values in the title column
    # NaNasYearAmount # Amount of NaN values in the year column
    # JoinedDfLength_afterNaNYearCleaning # Length of the dataframe after removing NaN values in the year column
    # MissingValues # Amount of missing values in the dataframe


    # # Save the Length and amount information to a dataframe and save it to a pickle file. The column name are the corresponding comments above.
    # LengthInfoDf=pd.DataFrame({"Directory Num": [dirNum],
    #                             "Length df": [JoinedDfLength],
    #                             "non unique titles": [NonUniTitleAmount],
    #                             "non unique dois": [NonUniDOIAmount],
    #                             "Length df after title cleaning": [JoinedDfLength_afterTitleCleaning],
    #                             "non unique dois after title cleaning": [NonUniDOIAmount_afterTitleCleaning],
    #                             "Length df after doi cleaning": [JoinedDfLength_afterDOICleaning],
    #                             "NaN title values": [NaNasTitleAmount],
    #                             "Length df after NaN title cleaning": [JoinedDfLength_afterNaNTitleCleaning],
    #                             "NaN year values": [NaNasYearAmount],
    #                             "Length df after NaN year cleaning": [JoinedDfLength_afterNaNYearCleaning],
    #                             "Total NaN in df": [MissingValues]})

    # Apply the CleanJoinedDf function to the dataframe
    LengthInfoDf, JoinedDf_Cleaned_4=CleanJoinedDf(JoinedDf, IntermediateData_Path, dirNum, DOIPath_Suffix)

    # Concatenate the LengthInfoDf with itself to create a dataframe with all the information if it is the first iteration
    if dirNum==StartDir:
        LengthInfoDf_All=LengthInfoDf
    else:
        LengthInfoDf_All=pd.concat([LengthInfoDf_All, LengthInfoDf], axis=0)

    # Save the joined and cleaned dataframe to a pickle file
    path=(IntermediateData_Path + str(dirNum).zfill(3) + MetaDataJoined_Suffix)
    JoinedDf_Cleaned_4.to_pickle(path)
    print(path, "is saved")

    print("Directory ", dirNum, "is done")

# Save the LengthInfoDf to a pickle file
path=(IntermediateData_Path + str(StartDir).zfill(3) + "_" + str(EndDir).zfill(3) + LengthInfoDf_Suffix)
LengthInfoDf_All.to_pickle(path)
print(path, "is saved")

# Save the LengthInfoDf to a csv file (without index)
path=(IntermediateData_Path + str(StartDir).zfill(3) + "_" + str(EndDir).zfill(3) + LengthInfoDf_Suffix2)
LengthInfoDf_All.to_csv(path, index=False)

-------------- # Check if RefDb has double entries

doi="10.1002/(sici)1099-1387(199907)5:7<313::aid-psc200>3.0.co;2-f"
# Find the row in the RefDf dataframe which has the same doi as the doi variable
RefDf[RefDf["doi"]==doi].compute()
#                                                       doi    year  is_oa publisher                journal_name            genre                                              title
# 1043    10.1002/(sici)1099-1387(199907)5:7<313::aid-ps...  1999.0  False     Wiley  Journal of Peptide Science  journal-article  Assembly of binding loops on aromatic template...
# 244097  10.1002/(sici)1099-1387(199907)5:7<313::aid-ps...  1999.0  False     Wiley  Journal of Peptide Science  journal-article  Assembly of binding loops on aromatic template...

--------------


# Test if the text files can be accessed
#----------------------------------------

# Load three of the of the saved joined and cleaned dataframes
path='Y:\\IntermediateData\\002_MetaDataJoined.pkl'
df5=pd.read_pickle(path)

import numpy as np

# Get the doi and data_dir value of the same random row  
random_index=np.random.randint(0,len(df5))
df5.iloc[random_index]
doi=df5.iloc[random_index]["doi"]
data_dir=df5.iloc[random_index]["data_dir"]

# Encode the doi string similar to URL encoding
# 'Y:\\Data\\00200000\\10.1002\\1521-4095(200110)13:20<1541::aid-adma1541>3.0.co;2-x.txt' ->  'Y:\\Data\\00200000\\10.1002\\1521-4095%28200110%2913%3A20%3C1541%3A%3Aaid-adma1541%3E3.0.co%3B2-x.txt'
import urllib.parse
doi_encoded=urllib.parse.quote(doi)
doi_encoded

# Create a path to the text file with the doi and data_dir value
path=("Y:\\Data\\" + str(data_dir).zfill(3) + "00000\\" + doi_encoded + ".txt").replace("/", "\\")
# Print the first 100 characters of the text file
with open(path, "r", encoding="utf8") as f:
    print(f.read(100))

# Load the file through loading the dictionary

# Load the _DOI_Path_Dict.pkl file
path='Y:\\IntermediateData\\002_DOI_Path_Dict.pkl'
DOI_Path_Dict=pd.read_pickle(path)
path2=DOI_Path_Dict[doi]

# Print the first 100 characters of the text file
with open(path2, "r", encoding="utf8") as f:
    print(f.read(300))

# Ft=open(path, "r", encoding="utf8").read()



#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Create dataframe for the Sankey Diagram
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
import dask.dataframe as dd
import plotly.graph_objects as go
import plotly.express as pex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


compressed_pdf_files # Amount of compressed pdf files for each directory
unzipping_and_pdf2text_loss # Amount of unzipping and pdf2text loss for each directory
text_files # Amount of text files for each directory
preprocessing_loss # Amount of preprocessing loss for each directory
preprocessed_files # Amount of preprocessed files for each directory
filtering_loss # Amount of filtering loss for each directory
filtered_files # Amount of filtered files for each directory
alignment_loss # Amount of alignment loss for each directory
aligned_files # Amount of aligned files for each directory
ref_db_files # Amount of entries in the reference database


# Define start and end directory
StartDir=0
EndDir=199

# Create an empty dataframe to store the information for each directory
SankeyDf=pd.DataFrame(columns=["dirNum","compressed_pdf_files", "unzipping_and_pdf2text_loss", "text_files", "preprocessing_loss", "preprocessed_files", "filtering_loss", "filtered_files", "alignment_loss", "aligned_files", "ref_db_files"])

# The amount of compressed pdf files for each directory is always 100000
compressed_pdf_files=100000

# The amount of ref_db_files can be calculated by checking the length of the xaa-xba.csv file (Read the file with Dask)
path='Y:\\Reference_Databases\\unpaywall\\xaa-xba.csv'
ref_db_files=len(dd.read_csv(path))

# Loop over the directories
for dirNum in range(StartDir, EndDir+1):

    # The amount of text files can be calulated by counting the number entries in the 000_DOI_Path_Dict.pkl dictionary
    path='Y:\\IntermediateData\\' + str(dirNum).zfill(3) +'_DOI_Path_Dict.pkl'
    DOI_Path_Dict=pd.read_pickle(path)
    text_files=len(DOI_Path_Dict)

    # The amount of unzipping_and_pdf2text_loss can be calculated by subtracting the amount of text_files from the amount of compressed_pdf_files
    unzipping_and_pdf2text_loss=compressed_pdf_files-text_files

    # The amount of preprocessed_files can be calculated by checking the length of the 000_MetaData.pkl dataframe
    path='Y:\\IntermediateData\\' + str(dirNum).zfill(3) +'_MetaData.pkl'
    MetaData=pd.read_pickle(path)
    preprocessed_files=len(MetaData)

    # The amount of preprocessing_loss can be calculated by subtracting the amount of preprocessed_files from the amount of text_files
    preprocessing_loss=text_files-preprocessed_files

    # The amount of filered_files can be calculated by checking the length of the 000_MetaData_Filtered.pkl dataframe
    path='Y:\\IntermediateData\\' + str(dirNum).zfill(3) +'_MetaDataFiltered.pkl'
    MetaData_Filtered=pd.read_pickle(path)
    filtered_files=len(MetaData_Filtered)

    # The amount of filtering_loss can be calculated by subtracting the amount of filtered_files from the amount of preprocessed_files
    filtering_loss=preprocessed_files-filtered_files

    # The amount of aligned_files can be calculated by checking the length of the 000_MetaDataJoined.pkl dataframe
    path='Y:\\IntermediateData\\' + str(dirNum).zfill(3) +'_MetaDataJoined.pkl'
    MetaDataJoined=pd.read_pickle(path)
    aligned_files=len(MetaDataJoined)

    # The amount of alignment_loss can be calculated by subtracting the amount of aligned_files from the amount of filtered_files
    alignment_loss=filtered_files-aligned_files

    # Append the information for each directory to the dataframe
    SankeyDf.loc[dirNum]=[dirNum, compressed_pdf_files, unzipping_and_pdf2text_loss, text_files, preprocessing_loss, preprocessed_files, filtering_loss, filtered_files, alignment_loss, aligned_files, ref_db_files]

# Save the dataframe to a csv file
SankeyDf.to_csv("Y:\\IntermediateData\\SankeyDf.csv")

# # Calculate the sum of each column in the SankeyDf except for the ref_db_lines and dirNum column and show it as a bar plot
# SankeyDf.sum(axis=0)[["compressed_pdf_files", "unzipping_and_pdf2text_loss", "text_files", "preprocessing_loss", "preprocessed_files", "filtering_loss", "filtered_files", "alignment_loss", "aligned_files"]].plot(kind="bar")
# # Make enough space in order that the x-axis labels are not cut off
# plt.tight_layout()
# # Show the plot
# plt.show()


# Calculate the sum of each column in the SankeyDf except for the ref_db_lines and dirNum column and assign it to a new dataframe
SankeyDfSum=SankeyDf.sum(axis=0)[["compressed_pdf_files", "unzipping_and_pdf2text_loss", "text_files", "preprocessing_loss", "preprocessed_files", "filtering_loss", "filtered_files", "alignment_loss", "aligned_files"]]

source_dest = [
    ["compressed_pdf_files", "text_files"], # 1
    ["compressed_pdf_files", "unzipping_and_pdf2text_loss"], # 2
    ["text_files", "preprocessed_files"], # 3
    ["text_files", "preprocessing_loss"], # 4
    ["preprocessed_files", "filtered_files"], # 5
    ["preprocessed_files", "filtering_loss"], # 6
    ["filtered_files", "Alignment_step"], # 7
    ["ref_db_files", "Alignment_step"], # 8
    ["Alignment_step", "aligned_files"], # 9
    ["Alignment_step", "alignment_loss"] # 10
]

SankePlotDf = pd.DataFrame(source_dest, columns=["Source", "Dest"])

#---------------------------------------------------1-------------------------------2-----------------------------------3-------------------------------------4-------------
SankePlotDf["Count"] = np.array([SankeyDfSum["text_files"],SankeyDfSum["unzipping_and_pdf2text_loss"],SankeyDfSum["preprocessed_files"],SankeyDfSum["preprocessing_loss"],

#--------------------5---------------------------6--------------------------------7-------------------------8----------------
SankeyDfSum["filtered_files"],SankeyDfSum["filtering_loss"], SankeyDfSum["filtered_files"], SankeyDf["ref_db_files"][0],

#--------------------9---------------------------10---------
SankeyDfSum["aligned_files"],SankeyDfSum["alignment_loss"]])


all_nodes =SankePlotDf.Source.values.tolist() + SankePlotDf.Dest.values.tolist()

source_indices = [all_nodes.index(source) for source in SankePlotDf.Source] ## Retrieve source nodes indexes as per all nodes list.
target_indices = [all_nodes.index(dest) for dest in SankePlotDf.Dest] ## Retrieve destination nodes indexes as per all nodes list.

colors = pex.colors.qualitative.D3

node_colors_mappings = dict([(node,np.random.choice(colors)) for node in all_nodes])

node_colors = [node_colors_mappings[node] for node in all_nodes]
edge_colors = [node_colors_mappings[node] for node in SankePlotDf.Source] ## Color links according to source nodes

fig = go.Figure(data=[
                    go.Sankey(
                        node = dict(
                                pad = 20,
                                thickness = 20,
                                line = dict(color = "black", width = 1.0),
                                label =  all_nodes,
                                color =  node_colors,
                               ),
                        link = dict(
                               source =  source_indices,
                               target =  target_indices,
                               value =  SankePlotDf.Count,
                               color = edge_colors
                               )
                         )
                    ])

fig.update_layout(title_text="SciView Sankey Diagram", font_size=16)
                  height=600,
                  font=dict(size = 10, color = 'white'),
                  plot_bgcolor='black', paper_bgcolor='black')

fig.show()



#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Check why some of path in the doiPathDictionary can not be read
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

import pickle

path='Y:\\IntermediateData\\000_DOI_Path_Dict.pkl'
doiPathDict=pickle.load(open(path, 'rb'))

with open(path, 'rb') as handle:
    interDict2 = pickle.load(handle)  
return interDict2

for doi, path in doiPathDict.items():

    # Load Text
    try:
        Ft=open(path, "r", encoding="utf8").read()

    
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Create and index for a dask dataframe in order to find values faster
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
import dask.dataframe as dd

# Load the dask dataframe
path='Y:\\Reference_Databases\\unpaywall\\xaa-xba.csv'
ref_db_files=len(dd.read_csv(path))

# Extract the doi column from the dask dataframe and use it to create an index
doi_index=dd.read_csv(path)["doi"].compute()

# Use a doi string to find the index of the doi in the doi_index
def find_doi_index(doi):
    return doi_index.get_loc(doi)








.



















#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Check if the language detection really works
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

df=pd.read_pickle("Y:\\IntermediateData\\000_MetaDataJoined.pkl")
df["language"].unique()
df[df["language"]=="no"]
list(df["doi"][df["language"]=="no"])[0]






















#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Concatenate all of the aligned and cleaned joined dataframes into a single one
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

import dask.dataframe as dd
import pandas as pd


# Define start and end directory numbers
StartDir=0
EndDir=2

# Define the suffixes for the joined and cleaned dataframes
MetaDataJoined_Suffix="_MetaDataJoined.pkl"

# Loop through the directories and concatenate the joined and cleaned dataframes
for dirNum in range(StartDir, EndDir+1):
    # Load the joined and cleaned dataframe
    path=(IntermediateData_Path + str(dirNum).zfill(3) + MetaDataJoined_Suffix)
    df=pd.read_pickle(path)

    # Append the dataframe to csv file and check that no additonal header row is added except for the first dataframe
    csvPath=(IntermediateData_Path + str(dirNum).zfill(3) + "_" + str(EndDir).zfill(3) + MetaDataJoined_Suffix2)

    if dirNum==StartDir:
        df_All.to_csv(csvPath, index=False)
    else:
        df_All.to_csv(csvPath, index=False, header=False)
    print(path, "is saved")


# Load the generated csv file into a dask dataframe
path=(IntermediateData_Path + str(StartDir).zfill(3) + "_" + str(EndDir).zfill(3) + ".csv")
df_All=dd.read_csv(path)

# Create a histogramm plot based on the year column in the df_All dask dataframe
df_All["year"].hist(bins=100).compute



































#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# SciView can only be activated through the powershell!!!!!!!!!!!!!!!!!!!!!
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# ## Streaming and sampling approach for phrase model, dictionary, BoW corpus and LDA
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Load packages
from Functions.F1_Subsets_and_PreProcessing import Dict_Loader
from Functions.F2_Model_Building import path_creator, log_printer, MyCorpus
import pickle
import pandas as pd
import gensim
import time
# import pyLDAvis.gensim_models
import gc
import memory_profiler as mem_profile
import csv
import ast

from modelConfig_0_99 import *
with open('modelConfig_0_99.py', 'r') as f:
    print(f.read())

# Define certain paths
LogfilePath=Model_Path + LogFileName
sampledDocTextFilePath=Model_Path + sampledDocPath

# Process through directories if an InterDir is defined it starts from there
for dirNum in range(StartDir,EndDir+1):

    # Load the Preprocess Meta Data of the current directory number
    Meta=pd.read_pickle(IntermediateData_Path + str(dirNum).zfill(3) + MetaDataFiltered_Suffix]))

    # Sample random entries of the Meta data by applying the pandas sample function to the meta data frame
    MetaSample=Meta.sample(n=sampleNDoc, random_state=randomState)

    # Load the preprocessed full text of the currend directory number
    FtPr=Dict_Loader(dirNum, IntermediateData_Path, FtPr_Suffix)

    # Filter the preprocessed full text based on the sampled meta data
    FtPrFi=[FtPr[DOI] for DOI in list(MetaSample["DOI"])]

    # Open a text file and write the sampled documents to it. The texte file is then used for streaming the sampled documents
    
    outfile= open(sampledDocTextFilePath,'a', encoding="utf-8")
    for doc in FtPrFi:
        outfile.writelines("%s " % token for token in doc)
        outfile.write('\n')
    outfile.close()

    # Append the MetaSample dataframe to the Metasample dataframe if it does not yet exist
    try:
        MetaSampleTotal=MetaSampleTotal.append(MetaSample,ignore_index=True)
        print("Appended")
    except:
        MetaSampleTotal=MetaSample
        print("Created")

    print("Directory with number: ", dirNum, " is sampled", file=open(LogfilePath,'a'))

print("-------------------------------------------------", file=open(LogfilePath,'a'))

# Get the number of lines in the sampled documents text file without loading the whole file
while open(sampledDocTextFilePath, "r") as f:
    for index, line in enumerate(f)

print("Length of the sampled Meta Data dataframe: ", len(MetaSampleTotal), " Length of the sampled doc text file: ", index, file=open(LogfilePath,'a'))

# Save the sampled MetaSampleTotal dataframe
MetaSampleTotal.to_pickle(Model_Path + PreProcessMetaDataSample_Suffix)

# Define the path for for the streaming class (MyCorpus())
CorpusStreamPath=sampledDocTextFilePath

# Intiate empty bigram model
bigram = gensim.models.phrases.Phrases(MyCorpus(), min_count=bigramMinFreq, threshold=bigramThreshold, max_vocab_size=phraseVocabSize)

# Freeze bigram model
frozen_bigram_model = bigram.freeze()

# Open a text file and write the sampled documents
outfile= open("test2bigram.txt",'a', encoding="utf-8")
for doc in frozen_bigram_model[MyCorpus()]:
    outfile.writelines("%s " % token for token in doc)
    outfile.write('\n')
outfile.close()


# Create the streaming class
class MyCorpus:
    def __iter__(self):
        for line in open('test2bigram.txt', encoding="utf-8"):
            # assume there's one document per line, tokens separated by whitespace
            yield line.lower().split()

#Intialize dictinary and add corpus with bigrams
dct=gensim.corpora.Dictionary(MyCorpus(),prune_at=dictVocab)

["DictAddDocs",dct.num_docs, len(dct),dct.num_pos]

# Create the streaming class for BoW Corpus


class MyCorpus:
    def __iter__(self):
        for line in open('test2bigram.txt', encoding="utf-8"):
            # assume there's one document per line, tokens separated by whitespace
            yield dct.doc2bow(line.lower().split())


import time
tic = time.perf_counter()
lda = gensim.models.ldamodel.LdaModel(corpus=MyCorpus(), 
                                      id2word=dct,
                                      num_topics=100,
                                      update_every=1,
                                      chunksize=10000,
                                      passes=1)
toc= time.perf_counter()
print("Time:", toc-tic)