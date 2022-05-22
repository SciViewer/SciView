# Load packages
from Functions.F1_Subsets_and_PreProcessing import Preprocessed_Dict_and_Metadata, Dict_Loader, Chunks
import pickle
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
# nltk.download('punkt')
# nltk.download('stopwords')
# stemmer = PorterStemmer()
# stop_words = stopwords.words('english') # or any other list of your choice
# lemmatizer = WordNetLemmatizer()
# pip install langdetect
# from langdetect import detect_langs
# import time
import multiprocessing as mp
# from itertools import islice

#----------------------------------------#
# If the conda environment does not get correctly activated (e.g. import gensim is not working)
# https://stackoverflow.com/questions/56623269/cmd-warning-python-interpreter-is-in-a-conda-environment-but-the-environment
#----------------------------------------#

# Define input paths and file names
IntermediateData_Path="Y:\\IntermediateData\\"
doiPath_Suffix="_DOI_Path_Dict.pkl"

# Define Output paths and file names
IntermediateData_Path="Y:\IntermediateData\ "
FtPr_Suffix="_FtPr.pkl"
MetaData_Suffix="_MetaData.pkl"
encodeError_Suffix="_errEnc.pkl"

# Set the dirs to save doi and paths
StartDir=42
EndDir=49

#----------------------------------------#

# Iterate trough data directories
for dirNum in range(StartDir,EndDir+1):
    #Load the dict
    dictItem=Dict_Loader(dirNum, IntermediateData_Path, doiPath_Suffix)  
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

    print("Preprocess the text files of dirNum: ", dirNum)

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
    FtPrName=(IntermediateData_Path + dirNum + FtPr_Suffix).replace(" ","")
    encErrName=(IntermediateData_Path + dirNum + encodeError_Suffix).replace(" ","")


    # Store the returned elements
    # create a binary pickle file 
    a = open(FtPrName,"wb")
    # write the python object (dict) to pickle file
    pickle.dump(FtPr,a)
    # close file
    a.close()

    #Try delete large dataframe (Prevent memory loss error)
    del(FtPr)

    # Store the returned elements
    # create a binary pickle file 
    b = open(encErrName,"wb")
    # write the python object (dict) to pickle file
    pickle.dump(encErr,b)
    # close file
    b.close()

    # Save dataframe of metaData
    metaData.to_pickle(metaDataName)

    print("saved all of the files")