# import memory_profiler as mem_profile


# def path_creator(pathType, ArgumentList):
#     if pathType=="log"or pathType=="metaSample" or pathType=="phraseModel" or pathType=="dictionary" or pathType=="ldaModel":
#         return (ArgumentList[0] + str(ArgumentList[1]).zfill(3) + "_" + str(ArgumentList[2]).zfill(3) + ArgumentList[3])
    
#     elif pathType=="meta":
#         return (ArgumentList[0] + str(ArgumentList[1]).zfill(3) + ArgumentList[2])

#     elif pathType=="bowCorpusDirectory":
#         return (ArgumentList[0] + str(ArgumentList[1]).zfill(3) + "_" + str(ArgumentList[2]).zfill(3)+ "_BowCorpusDatatest")

#     elif pathType=="phrasesInterSave":
#         return (ArgumentList[0] + str(ArgumentList[1]).zfill(3) + "_(" + str(ArgumentList[2]).zfill(3) + "_" + str(ArgumentList[3]).zfill(3) + ")" + ArgumentList[4])

#     # 0 IntermediateData_Path, 1 dirNum, 2 StartDir, 3 EndDir, 4 BowCorpus_Suffix
#     elif pathType=="bowCorpus":
#         return (ArgumentList[0] + str(ArgumentList[2]).zfill(3) + "_" + str(ArgumentList[3]).zfill(3)+ "_BowCorpusData\\"+
#         str(ArgumentList[1]).zfill(3) + "_(" + str(ArgumentList[2]).zfill(3) + "_" + str(ArgumentList[3]).zfill(3) + ")" + ArgumentList[4])


# def log_printer(logName, ArgumentList):

#     if ArgumentList[0]=="PhraseModelParameters":
#         print("Phrase Model Parameters: ", ArgumentList[1], ArgumentList[2], ArgumentList[3],file=open(logName,'a'))
#         print("Phrase Model Parameters: ", ArgumentList[1], ArgumentList[2], ArgumentList[3])

#     elif ArgumentList[0]=="MemoryUsage":
#         print(ArgumentList[1], "{} Mb".format(mem_profile.memory_usage()),file=open(logName,'a'))
#         print(ArgumentList[1], "{} Mb".format(mem_profile.memory_usage()))

#     elif ArgumentList[0]=="ProcessTime":
#         print(ArgumentList[1], ArgumentList[2], " is applied/loaded/processed... in seconds: ", round(ArgumentList[3],4), ", in minutes ", round((ArgumentList[3])/60,4), ", in hours: ", round((ArgumentList[3])/3600,4),file=open(logName,'a'))
#         print(ArgumentList[1], ArgumentList[2], " is applied/loaded/processed... in seconds: ", round(ArgumentList[3],4), ", in minutes ", round((ArgumentList[3])/60,4), ", in hours: ", round((ArgumentList[3])/3600,4))

#     elif ArgumentList[0]=="FinalProcessTime":
#         print(ArgumentList[1], " is loaded/processed... in seconds: ", round(ArgumentList[2],4), ", in minutes ", round((ArgumentList[2])/60,4), ", in hours: ", round((ArgumentList[2])/3600,4),file=open(logName,'a'))
#         print(ArgumentList[1], " is loaded/processed... in seconds: ", round(ArgumentList[2],4), ", in minutes ", round((ArgumentList[2])/60,4), ", in hours: ", round((ArgumentList[2])/3600,4))

#     elif ArgumentList[0]=="MetaFilter":
#         print("Num of Docs before/after filtering: ", ArgumentList[1], " / ", ArgumentList[2],file=open(logName,'a'))
#         print("Num of Docs before/after filtering: ", ArgumentList[1], " / ", ArgumentList[2])

#     elif ArgumentList[0]=="interLoad":
#         print("An intermediate saved phrase model is loaded from: ",ArgumentList[1],file=open(logName,'a'))
#         print("An intermediate saved phrase model is loaded from: ",ArgumentList[1])
#         print("-------------------------------------------------------",file=open(logName,'a'))
#         print("-------------------------------------------------------")

#     elif ArgumentList[0]=="BigramUpdate":
#         print("Updated Bigram Vocab is: ",ArgumentList[1],file=open(logName,'a'))
#         print("Updated Bigram Vocab is: ",ArgumentList[1])

#     elif ArgumentList[0]=="DictionaryParameters":
#         print("The dictionary has max vocab of: ",ArgumentList[1],file=open(logName,'a'))
#         print("The dictionary has max vocab of: ",ArgumentList[1])

#     elif ArgumentList[0]=="iterNext":
#         print("-------------------------------------------------------",file=open(logName,'a'))
#         print("-------------------------------------------------------")

#     elif ArgumentList[0]=="DictAddDocs":
#         print("Total number of docs added: ", ArgumentList[1], " Length of the dict: " ,ArgumentList[2], " Num of processed words: ", ArgumentList[3],file=open(logName,'a'))
#         print("Total number of docs added: ", ArgumentList[1], " Length of the dict: " ,ArgumentList[2], " Num of processed words: ", ArgumentList[3])


#     elif ArgumentList[0]=="DictFiltBefore":
#         print("Length of Dict before filtering: ", ArgumentList[1], " Filtering with min freq of: " ,ArgumentList[2],file=open(logName,'a'))
#         print("Length of Dict before filtering: ", ArgumentList[1], " Filtering with min freq of: " ,ArgumentList[2])

#     elif ArgumentList[0]=="DictFiltAfter":
#         print("Length of Dict after filtering: ", ArgumentList[1],file=open(logName,'a'))
#         print("Length of Dict after filtering: ", ArgumentList[1])

#     elif ArgumentList[0]=="newCorpusDoc":
#         print("BoW Corpus Length: ", ArgumentList[1]," Num of processed Documents: " ,ArgumentList[2],file=open(logName,'a'))
#         print("BoW Corpus Length: ", ArgumentList[1]," Num of processed Documents: " ,ArgumentList[2])

#     else:
#         print(ArgumentList[0], " Argument is not defined")


# # Create the streaming class
# class MyCorpus:
#     def __iter__(self):
#         for line in open(CorpusStreamPath, encoding="utf-8"):
#             # assume there's one document per line, tokens separated by whitespace
            # yield line.lower().split()