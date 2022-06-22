# Define input paths and file names
#-------------------------------------------
# Input paths
IntermediateData_Path="Y:\\IntermediateData\\"
FtPr_Suffix="_FtPr.pkl"
MetaData_Suffix="_MetaData.pkl"

# Define output path and file names
#-------------------------------------------
# Model_Path="Y:\\Models\\"
sampledDocPath="0_99_10000_sampledDocs.text"
sampledDocBigramPath="0_99_10000_sampledDocsBigram.text"

# Set the dirs to save doi and paths
StartDir=0
EndDir=99

# Sample Parameter
sampleNDocs=10000
randomState=42

# Logfile
LogFilePath="Y:\\Models\\0_99_10000_Sampling_Log.txt"

#------------------------------------------------------------------------------

# Define filters
minTokenN=200
maxTokenN=15000
language="en"

# Filtered meta data
MetaDataFiltered_Suffix="_MetaDataFiltered.pkl"
PreProcessMetaDataSamplePath="Y:\\Models\\0_99_10000_PreProcessMetaData.pkl"

#------------------------------------------------------------------------------

# Define Bigram/Phrase Parameters
bigramMinFreq=10
bigramThreshold=10
phraseVocabSize=400000000
# InterDir=25 # Else None

# Phrases Bigram
# InterSavePhrases_Suffix="_InterSaveBigram.pkl"
FreezedPhrasesPath="Y:\\Models\\0_99_10000_Sampling_FreezedBigram.pkl"
# FreezedPhrasesLog_Suffix="_FreezedBigramLog.txt"

#------------------------------------------------------------------------------

# Define Dictionary Parameters
dictVocab=60000000
filter_freq=10

# Dictionary
Dictionary_Suffix="_Dictionary"
# DictionaryLog_Suffix="_DictionaryLog.txt"
DictionaryFiltered_Suffix="_DictionaryFiltered"

#------------------------------------------------------------------------------

# #Bow Corpus
# BowCorpus_Suffix="_BowCorpus.mm"
# BowCorpusLog_Suffix="_BowCorpusLog.txt"

#------------------------------------------------------------------------------

# Define LDA Parameters
nTopics=50
updateEvery=1
chunkSize=10000
passes=1

# LDA
LDA_Suffix="_LDA.model"
LDALog_Suffix="_LDALog.txt"
LDAMulti_Suffix="_LDAMulti.model"
LDAMultiLog_Suffix="_LDAMultiLog.txt"








