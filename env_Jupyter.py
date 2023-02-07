#-----------------------------------------------------
#-----------------------------------------------------
###  Define input paths and file names ###
#-----------------------------------------------------
#-----------------------------------------------------

# Data directories
IntermediateData_Path="Y:\\IntermediateData\\"
DataPath="Y:\\Data\\"
ModelPath="Y:\\Models\\"

#-----------------------------------------------------
#-----------------------------------------------------
### Define output path and file names ###
#-----------------------------------------------------
#-----------------------------------------------------

## F1 Preprocessing of scientific article fulltext ##
#-----------------------------------------------------

# Dictionary with DOIs and corresponding Path
DOIPath_Suffix="_DOI_Path_Dict.pkl"

# Full Test Preprocessed
FtPr_Suffix="_FtPr.pkl"

# Meta Data from preprocessing step
MetaData_Suffix="_MetaData.pkl"

# Error logging from preprocessing step
EncodeError_Suffix="_errEnc.pkl"


## F2 Processing and alignment of unpaywall dataset ##
#-----------------------------------------------------

# Loging intervall when processing unpaywall dataset
logEntries=int(1e5) # Intervall of printing processing status variables

# Path to the unzipped and splitted unpaywall datasets
path='Y:\\Reference_Databases\\unpaywall\\splitted'

# Path to the concatenated csv unpaywall dataset file (from the splitted .pkl files)
csvPath='Y:\\Reference_Databases\\unpaywall\\xaa-xba.csv'

# Save path for the joined reference dataset and meta data from the preprocessing of fulltext
MetaDataJoined_Suffix="_MetaDataJoined.pkl"

# Path to directory designated to save files around dataset alignment and sankey plotting
AlignmentSankeyData_Path="Y:\\IntermediateData\\DatasetAlignmentSankey\\"

# File name suffix for storing processing status variables of the dataset alignment and cleaning of the dataframe
LengthInfoDf_Suffix="_LengthInfoDf.pkl"

# Same as above but as a readable csv file
LengthInfoDf_Suffix2="_LengthInfoDf2.csv"

# Save path for the ScieView dataset
SciViewDataset_Path='Y:\\SciViewDatasets\\SciView.csv'




## F3 Building Phrase Models, Dictionaries, Bow Corpus and LDA Models ##
#-----------------------------------------------------

# Phrase Model (Bigram) parameter
bigramMinFreq=10
bigramThreshold=10
phraseVocabSize=400000000

# Dictionary Parameter
sampleNDoc=100
randomState=1234
num_rows = 61892491
dictVocab=60000000

# Bigram Model save string
BigramPhraseModel_Suffix="BigramPhraseModel.pkl"

# Dictionary Model save string
DictionaryModel_Suffix="DictionaryModel.pkl"

# LDA Model save string
LDA_Suffix="LDA.model"





#--------- Cleaned up ------------

# Define input paths and file names
# IntermediateData_Path="Y:\\IntermediateData\\"
# MetaData_Suffix="_MetaData.pkl"
# MetaDataFiltered_Suffix="_MetaDataFiltered.pkl"


# # Define filters and corresponding suffix
# minTokenN=200
# maxTokenN=15000
# language="en"
# MetaDataFiltered_Suffix="_MetaDataFiltered.pkl"


# # Model_Path="Y:\\Models\\"
# sampledDocPath="0_99_10000_sampledDocs.text"
# sampledDocBigramPath="0_99_10000_sampledDocsBigram.text"

# # Set the dirs to save doi and paths
# StartDir=0
# EndDir=99

# # Sample Parameter
# sampleNDocs=10000
# randomState=42

# # Logfile
# LogFilePath="Y:\\Models\\0_99_10000_Sampling_Log.txt"

# #------------------------------------------------------------------------------

# # Filtered meta data

# PreProcessMetaDataSamplePath="Y:\\Models\\0_99_10000_PreProcessMetaData.pkl"

# #------------------------------------------------------------------------------

# # Define Bigram/Phrase Parameters
# bigramMinFreq=10
# bigramThreshold=10
# phraseVocabSize=400000000
# # InterDir=25 # Else None

# # Phrases Bigram
# # InterSavePhrases_Suffix="_InterSaveBigram.pkl"
# FreezedPhrasesPath="Y:\\Models\\0_99_10000_Sampling_FreezedBigram.pkl"
# # FreezedPhrasesLog_Suffix="_FreezedBigramLog.txt"

# #------------------------------------------------------------------------------

# # Define Dictionary Parameters
# dictVocab=60000000
# filter_freq=10

# # Dictionary
# Dictionary_Suffix="_Dictionary"
# # DictionaryLog_Suffix="_DictionaryLog.txt"
# DictionaryFiltered_Suffix="_DictionaryFiltered"

# #------------------------------------------------------------------------------

# # #Bow Corpus
# # BowCorpus_Suffix="_BowCorpus.mm"
# # BowCorpusLog_Suffix="_BowCorpusLog.txt"

# #------------------------------------------------------------------------------

# # Define LDA Parameters
# nTopics=50
# updateEvery=1
# chunkSize=10000
# passes=1

# # LDA
# LDA_Suffix="_LDA.model"
# LDALog_Suffix="_LDALog.txt"
# LDAMulti_Suffix="_LDAMulti.model"
# LDAMultiLog_Suffix="_LDAMultiLog.txt"









