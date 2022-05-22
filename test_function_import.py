def path_creator(pathType, prefix, startDir, endDir, suffix):
    if pathType=="log":
        return (prefix + str(startDir).zfill(3) + "_" + str(endDir).zfill(3) + suffix)


def log_printer(logName, ArgumentList):
    if ArgumentList[0]=="PhraseModelParameters":
        print("Phrase Model Parameters: ", ArgumentList[1], ArgumentList[2], ArgumentList[3],file=open(logName,'a'))
        print("Phrase Model Parameters: ", ArgumentList[1], ArgumentList[2], ArgumentList[3])

    elif ArgumentList[0]=="MemoryUsage":
        print("Memory(Before): {}Mb".format(mem_profile.memory_usage()))
        print("Memory(Before): {}Mb".format(mem_profile.memory_usage()))
    else:
        print(ArgumentList[0], " Argument is not defined")