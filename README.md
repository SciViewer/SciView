# 1) About SciView
This repository offers a  framework based on a Synology NAS system, WSL and Python software in odrer to dowload, process and visualize terabytes of fulltext scientific articles. WSL is utilized in order to download, transform and sort scientific publications and reference databases. Python is used in order enable NLP functionalities and textmining workflows.

The goal of this project can be described along the following points:
* Support the cause of [library genesis](https://en.wikipedia.org/wiki/Library_Genesis)
    * If you seek to support this or similar in the same manner be aware of any legal consequency which vary depending on ISP and country you live in
* Store and process terabytes of data with own hardware
* Create NLP models based on available data and reference databases
* Create a webbased visualzation tool for navigation millions of publications
* Write a blog about the insights and learning of this project

The repository itself consists of two different projects; the knowledge base describing the process and insights of this project and the actual repository for implementation.

## 1.1) SciView Framework
The actual framwork encompasses the following files:

* **README.md** -> Information about the repository, project and guide for set up with own hardware.
* **scidb** -> A collection of various function enabling the transformation of the downloaded publication data.
* **env** -> Path defined which are used in scidb functions
* **SciView.ipynb** -> Notebook about NLP workflows implemented with Python
* **modelConfig_0_99.py** -> Configuration file for generating models and corresponding data
* **test_envrionment.py** -> A bag of test code and other snippets


## 1. 2) Knowledge and Blog
As part this project several blog articles are written at it is best to start with the [Main section](/Knowledge%20and%20Blog/SciView_Main.md)

## 1.3) Imporant Notice
Read before continuation!

    Currently the directory and device naming should be followed in order successfully run the scidb commands. As of writing this, no functionality is implemented in order to automatically create the corresponding directories, settings on the NAS system and programming envrionment. So the following implementation steps should be seen as an example of how to implement the framework functionalities.

# 2) Setup of the scidb alias functions
1. Login as root
2. open the bash config

        nano ~/.bashrc

3. Paste the alias at the end of the bashrc file

        alias scidb='./scidb'

# 3) Directory structure
As mentioned above the directory structure should be followed because of some hardcoded paths and unflexible environment files

## 3.1) Setup guide

1. Open the synology NAS webinterface and the File Station program
2. Click on "Create" and select "Create New Shared Folder"
3. Name it "SciMagDir" and finish the setup
4. Create the following folders in the newly created shared folder
    * Data
    * InterdiateData
    * Models
    * RawData
    * Reference_Databases
    * Torrentfiles

## 3.2) Directory explanations and population example

* **Data**
    * Contains up to 100'000 text files in each subdirectory. Each subdirectory is based on the torrent identifier (000 -> 9XX). The corresponding pdf file (which has been transformed to a text file) are storred in the same corresponding directory before deletion.
* **IntermediateData**
    * This directory contains various data files indexed by the torrent identifier. Examples for such files is the NNN_FtPr.pkl file, which contains the tokenized and preprocessed text data (see text files in the Data directory) or the NNN_MetaDataFilered.pkl file which is a dataframe storing metadata from the preprocessing step.
* **Models**
    * Throughout the NLP model building various files are generated representing either models or data structures used for further applications.
* **RawData**
    * The compressed file resulting from the torrent download are stored in this directory. The content is meant to be unzipped and can be left for further seeding (e.g. torrent checks this location)
* **Reference_Databases**
    * The main reference databases currently used is the data dump from unpaywall
* **Torrentfiles**
    * Torrent files are managed in this directory

## 3.3) Directory structure and hyrarchy

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


# 4) Mounting of Synology network drive
In order to programmatically access the data the synology NAS has to be configured accordingly. The unix based commands access the data through mounting of the already, in the windows explorer, mounted network drive. Therefore the first step is it to mount the network drive in windows, also python code acesses the through this mount.

## 4.1) Windows network drive mount
Reference Link: [Create a network drive mount in windows](https://basic-tutorials.de/synology-netzlaufwerk-im-windows-explorer-einbinden/)

Prerequisite: NAS is already setup and directly interfaced through ehternet to the computer

1. Open windows explorer and right click at "Network" at the sidebar
2. Select "Map network drive..." and the window pops up
3. For drive: chose "Y:" in the dropdown menu (Default is "Y:")
4. Paste "\\SciMagDB\SciMagDir" into the Folder: box (Default is "\\SciMagDB\SciMagDir")
5. Tick "reconnect at sign-in"
6. Check if the folders on the NAS are mapped accordingly


## 4.2) Drive mount in WSL-2
Reference Link: [Mount network drive in WSL](https://docs.microsoft.com/en-us/archive/blogs/wsl/file-system-improvements-to-the-windows-subsystem-for-linux)

Prerequisite: The network folder is mounted at /mnt/share and therefore the "share" directory has to be first created.

The mounting of the windows network drive into WSL is done by applying

    scidb init

This command is part of the scidb commands and invoked automatically with other commands.

# 5) Download and intiate torrent files
The currently implemented function dowloads all torrent files from [Torrent Repo](http://libgen.rs/scimag/repository_torrent/) from the 0th to the 875th file and stores it in a specified folder.

## 5.1) Configuration of the download station application
Before adding torrents for download to the download station application the program has to be configured first

1. Open the synology web interface and open the "Download Station" application (Install if not available)
2. Click "Settings" symbol on the bottom left
3. Navigate to the "Location" tab on the sidebar and select the following path for Destination: "ScimagDir/RawData"
4. Navigate to the "Auto Extract" tab on the sidebar
5. Tick "Enable Auto Extract.." and chose "Extract to", also select the following path: "SciMagDir/Data"
6. Optional define a download schedule in the "General" tab at the sidebar


## 5.2) Download and start torrent files
In order to download

1. Check the torrent path in the env file (default: TORRENT_PATH=/mnt/share/Torrentfiles)
2. Execute

        scidb torrentdl

3. Open the synology web interface and navigate into the directory containing the torrent files
4. Select one or more torrent file, righ click and then select "Add to download station"
5. Check if the download has started in the "Download Station" Application
6. Continuously check if the files have been correctly donwloaded and extraceted to the above defined paths

## 5.3) Unzip the files manually
The scidb command suite offers a command for extraction of the zipped downloaded file which can be utilized like:

    scidb dlunzip 001 050

This command has to be executed as root user!


# 6) PDF to text
In order to transform the pdf content to text the "pdftotext" function from the poppler-utils library is used. This [link](https://www.cyberciti.biz/faq/converter-pdf-files-to-text-format-command/) describes the installation of the package and an example of published usage of this tools is given in this [article](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005962)

The following command example allows the transformation of pdf into a text file and mantaining the DOI as the name of the file itself

    scidb text2pdf 001 050

 # 7) Data storage management
 After tranforming the pdf into textfiles the most imporant part for further steps is done. Depending on the amount of available storage the pdf files and zipfiles can be removed, the following functions help to delete and clean up data.

 ## 7.1) Removing pdf files
After translating the pdf into text files the pdf files are not used anymore. They can then be removed with the following command:

    scidb rmpdf 000 050 

With the scidb overview command one can then check if all pdf files have been removed. Remaining pdf files can be manually deleted through a ".pdf" search in the file station application on the synology NAS webinterface.

## 7.2) Renaming textfiles
The tranlation of pdf into textfiles sometimes generates file endings such as ".pdf.text". This pattern can be detected after removing of pdf files and still see a lot of documents with the scidb overview function. In order to rename these files the following command can be applied:

    scidb rntext 000 050

## 7.3) Removing zip files
If a torrent is finished and will not be continued it can be deleted. After deletion of the torrent also the download zip files can be removed (usually after unzipping and tranlating pdf to text files). For zip file deletion the following command can be used:

    scidb rmzip 000 050


## 7.4) Important notice on deleting files on the NAS
By default a file which gets deleted on the NAS is transferred into the recyle bin (Recycle bin directory is autmatically generated in the shared NAS folder). So in order to gain unused storage the recycle bin has to be continuously emptied!

# 8) Download of reference database
Database dump from unpaywall.org (After filling a form a link is created)


    wget https://unpaywall-data-snapshots.s3.us-west-2.amazonaws.com/unpaywall_snapshot_2021-07-02T151134.jsonl.gz


# 9) Jupyter Notebook



