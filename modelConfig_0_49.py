# Define input paths and file names
#-------------------------------------------
# Input paths
IntermediateData_Path="Y:\\IntermediateData\\"
FtPr_Suffix="_FtPr.pkl"
MetaData_Suffix="_MetaData.pkl"

# Define output path and file names
#-------------------------------------------
Model_Path="Y:\\Models\\"

# Phrases Bigram
InterSavePhrases_Suffix="_InterSaveBigram.pkl"
FreezedPhrases_Suffix="_FreezedBigram.pkl"
FreezedPhrasesLog_Suffix="_FreezedBigramLog.txt"

# Filtered meta data
MetaDataFiltered_Suffix="_MetaDataFiltered.pkl"
MetaDataSample_Suffix="_MetaDataSample.pkl"

# Dictionary
Dictionary_Suffix="_Dictionary"
DictionaryLog_Suffix="_DictionaryLog.txt"
DictionaryFiltered_Suffix="_DictionaryFiltered"

#Bow Corpus
BowCorpus_Suffix="_BowCorpus.mm"
BowCorpusLog_Suffix="_BowCorpusLog.txt"

# LDA
LDA_Suffix="_LDA.model"
LDALog_Suffix="_LDALog.txt"
LDAMulti_Suffix="_LDAMulti.model"
LDAMultiLog_Suffix="_LDAMultiLog.txt"

# Define filters
minTokenN=200
maxTokenN=15000
language="en"

# Define Bigram/Phrase Parameters
bigramMinFreq=10
bigramThreshold=10
phraseVocabSize=400000000
InterDir=25 # Else None

# Define Dictionary Parameters
dictVocab=60000000
filter_freq=10

# Define LDA Parameters
nTopics=50

# Set the dirs to save doi and paths
StartDir=0
EndDir=49