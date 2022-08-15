# import dask.dataframe as dd
import pandas as pd
# import os
# import time
# import pickle

# Define the function to read the JoinedDf and return the filtered meta data and cleaned JoinedDf
def CleanJoinedDf(JoinedDf, dirNum):
    '''
    This function reads the JoinedDf and returns the filtered meta data and cleaned JoinedDf
        Input: JoinedDf, IntermediateData_Path, dirNum, DOIPath_Suffix
        Output: filtered meta data and cleaned JoinedDf
    '''
    
    # Get length of the joined dataframe
    JoinedDfLength=len(JoinedDf)
    
    # Geth the amount of non unique value in the "title" column
    nonUniTitleList=list(JoinedDf["title"].value_counts().index[JoinedDf["title"].value_counts()>1])
    NonUniTitleAmount=len(nonUniTitleList)
    # Get the amount of unique values in the "doi" column
    nonUniDoiList=list(JoinedDf["doi"].value_counts().index[JoinedDf["doi"].value_counts()>1])
    NonUniDOIAmount=len(nonUniDoiList)
    # Remove all of the rows in the JoinedDf with titles which are not unique 
    JoinedDf_Cleaned_1=JoinedDf[~JoinedDf["title"].isin(nonUniTitleList)]
    # Count the amount of rows in the cleaned dataframe
    JoinedDfLength_afterTitleCleaning=len(JoinedDf_Cleaned_1)

    # Check if after removing non unique titles if there are still non unique dois
    nonUniDoiList=list(JoinedDf_Cleaned_1["doi"].value_counts().index[JoinedDf_Cleaned_1["doi"].value_counts()>1])
    NonUniDOIAmount_afterTitleCleaning=len(nonUniDoiList)
    # Remove all of the rows in the JoinedDf with dois which are not unique 
    JoinedDf_Cleaned_2=JoinedDf_Cleaned_1[~JoinedDf_Cleaned_1["doi"].isin(nonUniDoiList)]
    # Count the amount of rows in the cleaned dataframe
    JoinedDfLength_afterDOICleaning=len(JoinedDf_Cleaned_2)

    # Get the amount on NaN values in the "title" column
    NaNasTitleList=list(JoinedDf_Cleaned_2["doi"][JoinedDf_Cleaned_2["title"].isna()==True])
    NaNasTitleAmount=len(NaNasTitleList)
    # Remove all of the rows in the JoinedDf with titles which are NaN
    JoinedDf_Cleaned_3=JoinedDf_Cleaned_2[~JoinedDf_Cleaned_2["doi"].isin(NaNasTitleList)]
    # Count the amount of rows in the cleaned dataframe
    JoinedDfLength_afterNaNTitleCleaning=len(JoinedDf_Cleaned_3)

    # Get the amount of NaN vlaues in the "year" column of the joined dataframe
    NaNasYearList=list(JoinedDf_Cleaned_3["doi"][JoinedDf_Cleaned_3["year"].isna()==True])
    NaNasYearAmount=len(NaNasYearList)
    # Remove all of the rows in the JoinedDf with years which are NaN
    JoinedDf_Cleaned_4=JoinedDf_Cleaned_3[~JoinedDf_Cleaned_3["doi"].isin(NaNasYearList)]
    # Count the amount of rows in the cleaned dataframe
    JoinedDfLength_afterNaNYearCleaning=len(JoinedDf_Cleaned_4)

    # Turn the year values in the "year" column into integers
    JoinedDf_Cleaned_4["year"]=JoinedDf_Cleaned_4["year"].astype(int)

    # Create a new column which contains the given dirNum in a three number format (e.g. 001) and add it to the dataframe
    JoinedDf_Cleaned_4["data_dir"]=str(dirNum).zfill(3)

    # Rename the "TokenAmount" column to "token_amount" and the "Language" column to "language"
    JoinedDf_Cleaned_4.rename(columns={"TokenAmount": "token_amount", "Language": "language"}, inplace=True)

    # Check each column for the amount of missing values
    MissingValues=JoinedDf_Cleaned_4.isna().sum().sum()

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


    # Save the Length and amount information to a dataframe and save it to a pickle file. The column name are the corresponding comments above.
    LengthInfoDf=pd.DataFrame({"Directory Num": [dirNum],
                                "Length df": [JoinedDfLength],
                                "non unique titles": [NonUniTitleAmount],
                                "non unique dois": [NonUniDOIAmount],
                                "Length df after title cleaning": [JoinedDfLength_afterTitleCleaning],
                                "non unique dois after title cleaning": [NonUniDOIAmount_afterTitleCleaning],
                                "Length df after doi cleaning": [JoinedDfLength_afterDOICleaning],
                                "NaN title values": [NaNasTitleAmount],
                                "Length df after NaN title cleaning": [JoinedDfLength_afterNaNTitleCleaning],
                                "NaN year values": [NaNasYearAmount],
                                "Length df after NaN year cleaning": [JoinedDfLength_afterNaNYearCleaning],
                                "Total NaN in df": [MissingValues]})

    return LengthInfoDf, JoinedDf_Cleaned_4