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
sampleNDoc=1000000
randomState=1234
num_rows = 61892491
dictVocab=60000000

# Bigram Model save string
BigramPhraseModel_Suffix="BigramPhraseModel.pkl"

# Dictionary Model save string
DictionaryModel_Suffix="DictionaryModel.pkl"

# LDA Model save string
LDA_Suffix="LDA.model"