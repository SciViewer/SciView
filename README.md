# SciView

<img src="Knowledge and Blog\P0_owl.png" width="300">

This repository offers a framework based on a Synology NAS system, WSL and Python in order to dowload, process and visualize terabytes of fulltext scientific articles. WSL is utilized to download, transform and sort scientific publications and reference databases. Python is used in a Jupyter Notebook to enable NLP functionalities and textmining workflows.

The goal of this project can be described along the following points:
* Support the cause of [Library Genesis](https://en.wikipedia.org/wiki/Library_Genesis) and [Unpaywall]()
    * If you seek to support this or similar in the same manner be aware of any legal consequency which vary depending on ISP and country you live in!
* Store and process terabytes of data with own hardware
* Create NLP models based on available data and reference databases
* Create a webbased visualzation tool for navigation millions of publications
* Write a blog about the insights and learning of this project

The repository itself consists of two different projects; the knowledge base (Blog articles) describing the process and insights of this project and the actual repository for implementation.

**Knowledge and Blog**

Accompanied to this repository blog articles are and will be written. The posts themself are also contained in this repository and the [**Main Section**](/Knowledge%20and%20Blog/SciView_Main.md) is the starting point of those posts.


**SciView Framework**

The actual framwork encompasses the following files:

* README.md
    * Information about the repository, project and guide for set up with own hardware.
* scidb
    * A collection of various function enabling the transformation of the downloaded publication data.
* env_scidb
    * Path defined which are used in scidb functions
* SciView.ipynb 
    * Notebook about NLP workflows
* env_scidb.py 
    * Configuration file for generating models and corresponding data
* test_envrionment.py
    * A bag of test codes and other snippets
* SciView_env.yaml
    * 
* Functions
    * 
* Knowledge and Blog
    * 

# SciView Dataset

The SciView dataset currently contains **61'892'491** rows, each referecing a scientific article with meta data such as the year of publishing or it's title. Furthermore with the doi and the corresponding directory number the full text content of the article can be accssed by constructing a link to the corresponding file.

This dataset was created through combining a dataset dump of over 138 million entries (see ref_db_files in picture below) of meta data of scholarly scientific articles and data repository of over 87 million full text scientific articles (see compressed_pdf_files in picture below).

The joining or alignment of these two datasets was done based on the digital object identifier (doi). The resulting dataset therefore enables access to over 61 million full text scholarly articles and the corresponding meta data.


<img src="Knowledge and Blog\P0_sankey.png" width="1400">

The dataset is a single csv file which is currently accessed through dask. In the picture below the head of the SciView dataset is shown with parameters such as doi, year of publishing or the genre of the article. Based on the data_dir and the doi a path to the fulltext content of the article can be generated.


<img src="Knowledge and Blog\P0_head.PNG" width="1200">


The tail of the dataset shows the last rows of the last dask dataframe partition (e.g. index of the last partition)

<img src="Knowledge and Blog\P0_tail.PNG" width="1200">

# Download of SciView dataset

While the dataset can be easily made accessible for download it is more difficult for the corresponding full text data as it contains open (26%) and non-open access (74%) articles. Nevertheless if the purpose of any further work utilizing this dataset and articles without sharing the underlying content of scientific articles (e.g. no copyright infringement), for example a text mining application, a download can be made accessible

For further requests get in contact through: sciview@protonmail.com


# Generate dataset and process with Jupyter Notebook
SciView is a collection of a dataset and corresponding analysis of it through NLP applications enabled in a Jupyter Notebook. 

The [**Guide to dataset Generation**](/Knowledge%20and%20Blog/Guide_Dataset_Generation.md) contains the following topics:
* Generating the Scieview dataset with functions and workflows enabled in Unix and Python
* An Overview on the content of Jupyter Notebook

# Setting up Python envrionment

An Anaconda Python 3.7.11 environment is set up and the installed package verrsions haven been exported to a yaml file with the following command

    conda env export > C:\Users\Public\Documents\SciView_env.yaml

The corresponding [**yaml**](SciView_env.yaml) file can be imported to in Anaconda. (It is recommended to use this enviroment. An example for possible problems with newer Python or package versions is the Pickle protocol version which can change depending on the version of Python.)
