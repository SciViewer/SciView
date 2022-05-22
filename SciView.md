# About
This repository offers a  framework based on a Synology NAS, WSL and Python in odrer to dowload, process and visualize terabytes of fulltext scientific articles.
# Setup of the scidb alias functions
1. Login as root
2. open the bash config

        nano ~/.bashrc

3. Paste the alias at the end of the bashrc file

        alias scidb='./scidb'

# Configure of Synology nas

# Directory structure

    /mnt/share=SciMagDir

    | SciMagDir
        | Data
            | 00000000
                | 10.1002
                    | file.pdf
                    | file.txt
                    | ... .pdf
                    | ... .txt
                | libgen.scimag00000000-00000999.zip
                | libgen.scimag00001000-00001999.zip                
            | 00100000
            | 00200000
            | ...
        | Torrentfiles
            | dowloaded
                |
                |
            |


# Download torrent files
The currently implemented function dowloads all torrent files from http://libgen.rs/scimag/repository_torrent/ from the 0th to the 851th file and stores it in a specified folder.
1. Check the torrent path in the env file
2. Execute

        scidb torrentdl

# Add torrent to Synology downloader


# Unzip the files

# PDF to text
The pdftotext function from the poppler-utils library is used. The library isreferenced here:

* https://www.cyberciti.biz/faq/converter-pdf-files-to-text-format-command/

* https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005962

