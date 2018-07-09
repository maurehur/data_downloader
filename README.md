# data_downloader

dado (**DataDownloader**) download and process the most popular Machine Learning datasets.

```
dado(dataset, 
    rm=False
    )
```

Download and process datasets in a directory called 'data'

Args:

* dataset: A string, the name of a dataset.
* rm: if False (default) keeps the files (.zip, .idx, .gz) downloaded. if True, remove the files (.zip, .idx, .gz) located at 'data' folder. 


## Usage

Instructions:

git clone https://github.com/maurehur/data_downloader.git

```
from data_downloader.get import dado
	
data = dado('udacity-dataset', rm=True)

###################################################
Attempting to download: udacity-dataset
###################################################
- Opening url: https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip
- Progress: 100%
- Download Complete!
- Found and verified: udacity-dataset.zip
- Extracting udacity-dataset.zip
- Moving Files and built csv
- Removing udacity-dataset.zip and unzip_folder
```

The available datasets are:

```
mnist
german-traffic-signs
udacity-dataset
```

