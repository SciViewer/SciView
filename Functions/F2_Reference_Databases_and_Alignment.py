# import dask.dataframe as dd
import pandas as pd
# import dask.dataframe as dd
# import os
# import time
# import pickle
import plotly.graph_objects as go
import plotly.express as pex
# import matplotlib.pyplot as plt
import numpy as np

# Define the function to read the JoinedDf and return the filtered meta data and cleaned JoinedDf
def Clean_Joined_Df(JoinedDf, dirNum):
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




def Sankey_Dataframe(StartDir, EndDir, ref_db_files):
    '''
    This function get a start dir and an end dir and returns a dataframe containing values for plotting a sankey digaram
        Input: StartDir, EndDir, compressed_pdf_files, ref_db_files
        Output: filtered meta data and cleaned JoinedDf
    '''
    # Create an empty dataframe to store the information for each directory
    SankeyDf=pd.DataFrame(columns=["dirNum","compressed_pdf_files", "unzipping_and_pdf2text_loss", "text_files", "preprocessing_loss",
                             "preprocessed_files", "alignment_loss", "aligned_files", "ref_db_files"])

    # The amount of compressed pdf files for each directory is always 100000
    compressed_pdf_files=100000


        # Loop over the directories
    for dirNum in range(StartDir, EndDir+1):

        # The amount of text files can be calulated by counting the number entries in the 000_DOI_Path_Dict.pkl dictionary
        path='Y:\\IntermediateData\\' + str(dirNum).zfill(3) +'_DOI_Path_Dict.pkl'
        DOI_Path_Dict=pd.read_pickle(path)
        text_files=len(DOI_Path_Dict)

        # The amount of unzipping_and_pdf2text_loss can be calculated by subtracting the amount of text_files from the amount of compressed_pdf_files
        unzipping_and_pdf2text_loss=compressed_pdf_files-text_files

        # The amount of preprocessed_files can be calculated by checking the length of the 000_MetaData.pkl dataframe
        path='Y:\\IntermediateData\\' + str(dirNum).zfill(3) +'_MetaData.pkl'
        MetaData=pd.read_pickle(path)
        preprocessed_files=len(MetaData)

        # The amount of preprocessing_loss can be calculated by subtracting the amount of preprocessed_files from the amount of text_files
        preprocessing_loss=text_files-preprocessed_files

        # # The amount of filered_files can be calculated by checking the length of the 000_MetaData_Filtered.pkl dataframe
        # path='Y:\\IntermediateData\\' + str(dirNum).zfill(3) +'_MetaDataFiltered.pkl'
        # MetaData_Filtered=pd.read_pickle(path)
        # filtered_files=len(MetaData_Filtered)

        # # The amount of filtering_loss can be calculated by subtracting the amount of filtered_files from the amount of preprocessed_files
        # filtering_loss=preprocessed_files-filtered_files

        # The amount of aligned_files can be calculated by checking the length of the 000_MetaDataJoined.pkl dataframe
        path='Y:\\IntermediateData\\' + str(dirNum).zfill(3) +'_MetaDataJoined.pkl'
        MetaDataJoined=pd.read_pickle(path)
        aligned_files=len(MetaDataJoined)

        # The amount of alignment_loss can be calculated by subtracting the amount of aligned_files from the amount of filtered_files
        alignment_loss=preprocessed_files-aligned_files

        # # Append the information for each directory to the dataframe
        # SankeyDf.loc[dirNum]=[dirNum, compressed_pdf_files, unzipping_and_pdf2text_loss, text_files, preprocessing_loss, preprocessed_files, filtering_loss, filtered_files, alignment_loss, aligned_files, ref_db_files]

        # Append the information for each directory to the dataframe
        SankeyDf.loc[dirNum]=[dirNum, compressed_pdf_files, unzipping_and_pdf2text_loss, text_files, preprocessing_loss, preprocessed_files, alignment_loss, aligned_files, ref_db_files]

    return SankeyDf



def Sankey_DataFlow_Graph(SankeyDf):

    # Calculate the sum of each column in the SankeyDf except for the ref_db_lines and dirNum column and assign it to a new dataframe
    SankeyDfSum=SankeyDf.sum(axis=0)[["compressed_pdf_files", "unzipping_and_pdf2text_loss", "text_files", "preprocessing_loss",
                    "preprocessed_files", "alignment_loss", "aligned_files"]]

    # source_dest = [
    #     ["compressed_pdf_files", "text_files"], # 1
    #     ["compressed_pdf_files", "unzipping_and_pdf2text_loss"], # 2
    #     ["text_files", "preprocessed_files"], # 3
    #     ["text_files", "preprocessing_loss"], # 4
    #     ["preprocessed_files", "filtered_files"], # 5
    #     ["preprocessed_files", "filtering_loss"], # 6
    #     ["filtered_files", "Alignment_step"], # 7
    #     ["ref_db_files", "Alignment_step"], # 8
    #     ["Alignment_step", "aligned_files"], # 9
    #     ["Alignment_step", "alignment_loss"] # 10
    # ]

    source_dest = [
        ["compressed_pdf_files", "text_files"], # 1
        ["compressed_pdf_files", "unzipping_and_pdf2text_loss"], # 2
        ["text_files", "preprocessed_files"], # 3
        ["text_files", "preprocessing_loss"], # 4
        ["preprocessed_files", "Alignment_step"], # 5
        ["ref_db_files", "Alignment_step"], # 8
        ["Alignment_step", "aligned_files"], # 9
        ["Alignment_step", "alignment_loss"] # 10
    ]


    # SankePlotDf = pd.DataFrame(source_dest, columns=["Source", "Dest"])
    # #---------------------------------------------------1-------------------------------2-----------------------------------3-------------------------------------4-------------
    # SankePlotDf["Count"] = np.array([SankeyDfSum["text_files"],SankeyDfSum["unzipping_and_pdf2text_loss"],SankeyDfSum["preprocessed_files"],SankeyDfSum["preprocessing_loss"],
    # #--------------------5---------------------------6--------------------------------7-------------------------8----------------
    # SankeyDfSum["filtered_files"],SankeyDfSum["filtering_loss"], SankeyDfSum["filtered_files"], SankeyDf["ref_db_files"][0],
    # #--------------------9---------------------------10---------
    # SankeyDfSum["aligned_files"],SankeyDfSum["alignment_loss"]])


    SankePlotDf = pd.DataFrame(source_dest, columns=["Source", "Dest"])

    SankePlotDf["Count"] = np.array([SankeyDfSum["text_files"], # 1
                                    SankeyDfSum["unzipping_and_pdf2text_loss"], # 2
                                    SankeyDfSum["preprocessed_files"], # 3
                                    SankeyDfSum["preprocessing_loss"], # 4
                                    SankeyDfSum["preprocessed_files"], # 5
                                    SankeyDf["ref_db_files"][0], # 8
                                    SankeyDfSum["aligned_files"], # 9
                                    SankeyDfSum["alignment_loss"]]) # 10


    all_nodes =SankePlotDf.Source.values.tolist() + SankePlotDf.Dest.values.tolist()

    source_indices = [all_nodes.index(source) for source in SankePlotDf.Source] ## Retrieve source nodes indexes as per all nodes list.
    target_indices = [all_nodes.index(dest) for dest in SankePlotDf.Dest] ## Retrieve destination nodes indexes as per all nodes list.

    # colors = pex.colors.qualitative.D3
    # node_colors_mappings = dict([(node,np.random.choice(colors)) for node in all_nodes])
    # node_colors = [node_colors_mappings[node] for node in all_nodes]
    # edge_colors = [node_colors_mappings[node] for node in SankePlotDf.Source] ## Color links according to source nodes

    # The node colors have to be set to 16 labels. The ligned files labelis highlighted with another color
    # ['compressed_pdf_files', 'compressed_pdf_files', 'text_files', 'text_files', 'preprocessed_files', 'ref_db_files', 'Alignment_step', 'Alignment_step',
    #  'text_files', 'unzipping_and_pdf2text_loss', 'preprocessed_files', 'preprocessing_loss', 'Alignment_step', 'Alignment_step', 'aligned_files', 'alignment_loss']
    colors=pex.colors.sequential.Blugrn
    node_colors=14*[colors[6]] 
    node_colors.append("rgb(255, 238, 136)")
    node_colors.append(colors[6])

    # The edge colors have to be set to 8 labels with a gradient color pattern
    # ['compressed_pdf_files', 'compressed_pdf_files', 'text_files', 'text_files', 'preprocessed_files', 'ref_db_files', 'Alignment_step', 'Alignment_step']
    colors=pex.colors.sequential.tempo
    edge_colors=colors[-8::]

    fig = go.Figure(data=[
                        go.Sankey(
                            node = dict(
                                    pad = 20,
                                    thickness = 20,
                                    line = dict(color = "black", width = 1.0),
                                    label =  all_nodes,
                                    color =  node_colors,
                                ),
                            link = dict(
                                source =  source_indices,
                                target =  target_indices,
                                value =  SankePlotDf.Count,
                                color = edge_colors
                                )
                            )
                        ])

    fig.update_layout(title_text="SciView Sankey Diagram",
                    height=600,
                    font=dict(size = 12, color = 'black'),
                    plot_bgcolor='black', paper_bgcolor='white')
    return fig



