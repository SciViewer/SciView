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
* env
    * Path defined which are used in scidb functions
* SciView.ipynb 
    * Notebook about NLP workflows
* modelConfig_0_99.py 
    * Configuration file for generating models and corresponding data
* test_envrionment.py
    * A bag of test codes and other snippets

# Generate dataset and process Jupyter Notebook
SciView is a collection of a dataset and corresponding analysis of it through NLP applications enabled in a Jupyter Notebook. 

The [**Guide to dataset Generation**](/Knowledge%20and%20Blog/Guide_Dataset_Generation.md) contains the following topics:
* Generating the Scieview dataset with functions and workflows enabled in Unix and Python
* An Overview on the content of Jupyter Notebook

# ToDo
[ ] Clean up Jupyter Notebook
[ ] Comment all of the functions
[ ] Finish Chapter 3 then
