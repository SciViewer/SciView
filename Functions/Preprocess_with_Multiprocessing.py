# Load packages
from Functions.F1_Subsets_and_PreProcessing import Preprocessed_Dict_and_Metadata, Dict_Loader, Chunks
import pickle
import pandas as pd
import multiprocessing as mp
# import gc
import time

#----------------------------------------#
# If the conda environment does not get correctly activated (e.g. import gensim is not working)
# https://stackoverflow.com/questions/56623269/cmd-warning-python-interpreter-is-in-a-conda-environment-but-the-environment
#----------------------------------------#

from env_Jupyter import *
# with open('env_Jupyter.py', 'r') as f:
#     print(f.read())

# Set the dirs to save doi and paths
StartDir=100
EndDir=100

#----------------------------------------#

# Iterate trough data directories
for dirNum in range(StartDir,EndDir+1):

    tic = time.perf_counter()

    #Load the dict
    dictItem=Dict_Loader(dirNum, IntermediateData_Path, DOIPath_Suffix)  
    print("Length of dictionary num:", dirNum, "is", len(dictItem),"First two keys are:", list(dictItem.keys())[0:2])

    # Init a list which slices the dictionary into multiple dictionaries (each a chunk af 10000)
    slicedDictList=[]
    # Create and append dictionary chunks
    for item in Chunks(dictItem, 10000):
        slicedDictList.append([item,IntermediateData_Path,dirNum])
        print("Length of the slice is:",len(item), "First two keys of the slice are:", list(item.keys())[0:2])

    # Process each dictionary chunk
    print("Available cores: ",mp.cpu_count(), "(Pool = amount of cores)")
    pool = mp.Pool(processes=10)
    print("pool with 10 processes")
    Return=pool.map(Preprocessed_Dict_and_Metadata,slicedDictList)
    pool.close
    print("Preprocessed the text files of dirNum: ", dirNum)

    # Append Metadata
    slicedMetaDataList=[]
    for item in Return:
        slicedMetaDataList.append(item[0])
    metaData = pd.concat(slicedMetaDataList)

    # Append encoidng error dicitonaries
    encErr={}
    for item in Return:
        encErr.update(item[1])

    # Create name for the metaData
    # Bring for example 27 into the form of "027"
    dirNum=str(dirNum).zfill(3)
    # Create path to dictionary
    metaDataName=(IntermediateData_Path + dirNum + MetaData_Suffix).replace(" ","")
    encErrName=(IntermediateData_Path + dirNum + EncodeError_Suffix).replace(" ","")

    # Store the returned elements
    # create a binary pickle file 
    b = open(encErrName,"wb")
    # write the python object (dict) to pickle file
    pickle.dump(encErr,b)
    # close file
    b.close()

    # Save dataframe of metaData
    metaData.to_pickle(metaDataName)

    toc = time.perf_counter()
    print("Processing of dirNum: ", dirNum, " took: ", (toc-tic)/60, " minutes") 
