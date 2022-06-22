# About
This repository offers a  framework based on a Synology NAS, WSL and Python in odrer to dowload, process and visualize terabytes of fulltext scientific articles. WSL is utilized in order to download, transform and sort scientific publications and reference databases. The actual NLP functionalities are applied with Python.

This Article explains the 

# Setup of the scidb alias functions
1. Login as root
2. open the bash config

        nano ~/.bashrc

3. Paste the alias at the end of the bashrc file

        alias scidb='./scidb'



# Directory structure
## Setup guide

1.
2.
3.

## Directory explanations and population example

**Data**

Contains up to 100'000 text files in each subdirectory. Each subdirectory is based on the torrent identifier (000 -> 9XX). The corresponding pdf file (which has been transformed to a text file) are storred in the same corresponding directory before deletion.

**IntermediateData**

This directory contains various data files indexed by the torrent identifier. Examples for such files is the NNN_FtPr.pkl file, which contains the tokenized and preprocessed text data (see text files in the Data directory) or the NNN_MetaDataFilered.pkl file which is a dataframe storing metadata from the preprocessing step.

**Models**

Throughout the NLP model building various files are generated representing either models or data

The remote folder has to be defined on the synology NAS as "/SciMagDB/SciMagDir", the folder is then mounted as:

    /mnt/share=SciMagDir/

Subdirectories will be set and populated as follows: 

    Data
        00000000
        00100000
        00200000
            10.1002
            10.1016
               0009-2614%2891%2990484-q
               0003-2670%2894%2985094-1
               ...
    IntermediateData
        000_errEnc.pkl
        000_FtPr.pkl
        000_MetaData.pkl
        000_MetaDataFiltered.pkl
        001_DOI_Path_Dict.pkl
        ...
    Models
        000_049_LDA.model
        000_049_LDALog
        000_049_Dictionary
        000_049_Dictionary
        000_049_FreezedBigram.pkl
        ...
    RawData
        30000000
        30100000
        30200000
            libgen.scimag30200000-30200999
            libgen.scimag30201000-30201999
            libgen.scimag30202000-30202999
    Reference_Databases
        libgen_scimag
        scihub
        unpaywall
            unpaywall_snapshot_2021-07-02T151134.jsonl.gz
            unpaywall_snapshot_2021-07-02T151134.jsonl
            xaa
            xaa.pkl
            xab
            xab.pkl
            ...
    Torrenfiles
        sm_00000000-00099999
        sm_00100000-00199999
        sm_00200000-00299999
        ...


# Mounting of Synology network drive
In order to programmatically access the data the synology NAS has to be configured accordingly. The unix based commands access the data through mounting of the already, in the windows explorer, mounted network drive. Therefore the first step is it to mount the network drive in windows, also python code acesses the through this mount.
## Windows network drive mount
Reference Link: https://basic-tutorials.de/synology-netzlaufwerk-im-windows-explorer-einbinden/

Prerequisite: NAS is already setup and directly interfaced through ehternet to the computer

1. Open windows explorer and right click at "Network" at the sidebar
2. Select "Map network drive..." and the window pops up
3. For drive: chose "Y:" in the dropdown menu (Default is "Y:")
4. Paste "\\SciMagDB\SciMagDir" into the Folder: box (Default is "\\SciMagDB\SciMagDir")
5. Tick "reconnect at sign-in"
6. Check if the folders on the NAS are mapped accordingly


## Drive mount in WSL-2
Reference Link: https://docs.microsoft.com/en-us/archive/blogs/wsl/file-system-improvements-to-the-windows-subsystem-for-linux

Prerequisite: The network folder is mounted at /mnt/share and therefore the "share" directory has to be first created.

The mounting of the windows network drive into WSL is done by applying

    scidb init

This command is part of the scidb commands and invoked automatically with other commands.

# Download and intiate torrent files
The currently implemented function dowloads all torrent files from http://libgen.rs/scimag/repository_torrent/ from the 0th to the 875th file and stores it in a specified folder.

## Configuration of the download station application
Before adding torrents for download to the download station application the program has to be configured first

1. Open the synology web interface and open the "Download Station" application (Install if not available)
2. Click "Settings" symbol on the bottom left
3. Navigate to the "Location" tab on the sidebar and select the following path for Destination: "ScimagDir/RawData"
4. Navigate to the "Auto Extract" tab on the sidebar
5. Tick "Enable Auto Extract.." and chose "Extract to", also select the following path: "SciMagDir/Data"
6. Optional define a download schedule in the "General" tab at the sidebar


## Download and start torrent files
In order to download

1. Check the torrent path in the env file (default: TORRENT_PATH=/mnt/share/Torrentfiles)
2. Execute

        scidb torrentdl

3. Open the synology web interface and navigate into the directory containing the torrent files
4. Select one or more torrent file, righ click and then select "Add to download station"
5. Check if the download has started in the "Download Station" Application
6. Continuously check if the files have been correctly donwloaded and extraceted to the above defined paths

## Unzip the files manually
The scidb command suite offers a command for extraction of the zipped downloaded file which can be utilized like:

    scidb dlunzip 001 050

This command has to be executed as root user!


# PDF to text
The pdftotext function from the poppler-utils library is used. The library isreferenced here:

* https://www.cyberciti.biz/faq/converter-pdf-files-to-text-format-command/

* https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005962


# Download of reference database
Database dump from unpaywall.org (After filling a form a link is created)


    wget https://unpaywall-data-snapshots.s3.us-west-2.amazonaws.com/unpaywall_snapshot_2021-07-02T151134.jsonl.gz
