# Data Storage with a Synology NAS
Data storage has never been cheaper and more accessible than now, so why even bother with managing your own data storage soulution and hardware? The arguments for and against a selfowned NAS solution have already been discussed in other [sources](https://www.goodcloudstorage.net/cloud-research/nas-vs-cloud-storage/), I therefore want to highlight my own reasoning behind the data storage solution I chose in the context of managing a hughe torrent library.

## Why even selfowned storage?
If one thinks about trying to process a huge amount of data the go to solutions available today are clear. An arsenal of cloud storage and data processing for your specific need is available through a range of suppliers. So why store data on your own? The reasoning I followed were mainly based on the following points:
- Part of this projects success is it to support a torrent repository. In order to protect the data around and not rely on legal requirements of other parties and countries
- Setting up and solving technical issues around a NAS device was something I lacked and was interested to solve on my own
- Feasability of using customer hardware. I also wante to find out if I am able to handle and process terabytes of data with my computational and storage hardware.
The next few sections show specific solutions or settings around setting up a synology NAS.

## Part 1 - Guide 1 - Choosing a NAS
This subsjective guide shows how I went through the buying and expansion process of my NAS solution and should propose some chunks for though around choosing a NAS system.

### NAS for testing
The first NAS for testing purposes was a Synology DS118 with a 1 TB harddrive. The brand Synology was chosen based on its widespread use for home storage solutions.

### NAS for scaling up and hosting the data
For scaling up I calculated that the whole repository will be around 56 TB (more to it on learning and pitfalls). I got hands on a secondhand DS1813+ offering eight bays. I gradually equipped it with 8 TB harddrives.

### NAS for backup
Backing part intermediate processed data was expected to arount the amount of 8 TB.

### Learnings and pitfalls
- I was not aware of the different RAID configuration of a multi disk NAS. I therefore went with the standard Synology hybrid raid (SHR). After inserting the second 8TB the storage did not expand as expected. After some googling was clear the the SHR configuration uses one harddrive as backup drive and therfore needs a third drive for the first expansion of the total volume.
- Only go with one type of HD size is preferred. There comes a lot of additional complexity when working with different sized HDs. Just stick with one size makes it a lot easier.
- 


## Part 1 - Guide 2 - Interfaces to the NAS
I had the following 
- Web application of the synology
- Mounted directory in WSL (ubuntu)
- Network directory in Windows for Python (Anaconda)
- SSH

### Synology web application


