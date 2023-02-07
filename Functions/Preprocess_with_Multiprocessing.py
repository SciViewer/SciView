# Load packages
from Functions.F1_Subsets_and_PreProcessing import Preprocessed_Dict_and_Metadata, Dict_Loader, Chunks, Get_DOI_Prefix
import pickle
import pandas as pd
import multiprocessing as mp
# import gc
import os
import time

#----------------------------------------#
# If the conda environment does not get correctly activated (e.g. import gensim is not working)
# https://stackoverflow.com/questions/56623269/cmd-warning-python-interpreter-is-in-a-conda-environment-but-the-environment
#----------------------------------------#

from env_Jupyter import *
# with open('env_Jupyter.py', 'r') as f:
#     print(f.read())

# Set the dirs to save doi and paths
StartDir=200
EndDir=399



#----------------------------------------#

# Iterate trough data directories
for dirNum in range(StartDir,EndDir+1):

    tic = time.perf_counter()

    #Load the dict 
    dictItem=Dict_Loader(dirNum, IntermediateData_Path, DOIPath_Suffix)  
    print("Length of dictionary num:", dirNum, "is", len(dictItem),"First two keys are:", list(dictItem.keys())[0:2])


    # Get a list of unique Doi Prefixes in order to create the corresponding directories
    # doiPathDict=Dict_Loader(dirNum, IntermediateData_Path, DOIPath_Suffix) 
    doiPathDf=pd.DataFrame(dictItem.values())
    doiPrefixUniqueList=list(doiPathDf[0].apply(Get_DOI_Prefix).unique())


    # Create a new folder with the directory number as the name under the IntermediateData_Path
    dirNumPath=IntermediateData_Path + str(dirNum).zfill(3) + "\\"
    if os.path.exists(dirNumPath):
        None
    else:
        os.mkdir(dirNumPath)
        print(dirNumPath," path did not exist and has been created")

    # Create directory for every doi prefix
    for prefix in doiPrefixUniqueList:
        dirNumPrefixPath= IntermediateData_Path + str(dirNum).zfill(3) + "\\" + prefix
        if os.path.exists(dirNumPrefixPath):
            None
        else:
            os.mkdir(dirNumPrefixPath)
            print(dirNumPrefixPath," path did not exist and has been created")

#----------------
#----------------
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

#----------------
#----------------
    # # Setup without multiprocessing
    # DictList=[dictItem,IntermediateData_Path,dirNum]
    # Return=Preprocessed_Dict_and_Metadata(DictList)
#----------------
#----------------

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
