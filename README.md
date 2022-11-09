Attribute Extraction from Web Documents.

# Title: "Simplified DOM Trees for Transferable Attribute Extraction from the Web"

## Original paper
The conference paper is available via **[here](https://arxiv.org/pdf/2101.02415.pdf)**.

## Keywords
structured data extraction, web information extraction, Simplified DOM

## Implementation details
The implementation is in **[PyTorch](https://pytorch.org/)**.

Make sure that all Notebooks use **Python3 kernel.**

# Pre-trained model
Trained weights on SWDE dataset (**auto**- vertical) are available **[here](https://drive.google.com/file/d/1aMuHb8RT_GrKr6VoUvmDsObEwIqypkHy/view?usp=sharing)**.

In order to execute **test.ipynb** notebook, download the file and unzip in `./data` folder.

To run the pre-trained model Modify and execute the **test.ipynb** notebook:
- Set the `test_websites` and `attributes`, based on the content of `SWDE_Dataset/webpages/` and the description found in `SWDE_Dataset/readme.txt` 
- Execute the notebook

## To re-train the model
For that one has to download the **[SWDE dataset](https://academictorrents.com/details/411576c7e80787e4b40452360f5f24acba9b5159)**:

- Download the SWDE dataset via the Torrent file found on the references webpage
- Choose the vertical you want to train the model on the list of verticals in `SWDE_Dataset/webpages/`
- Extract the vertical folder into the `./data` subfolder
   - For instance: `SWDE_Dataset/webpages/movies.7z` into `./data/movies`
- Extract the ground trooth folder into the *./data` subfolder
   - For example: `SWDE_Dataset/groundtruth.7z` into `./data/groundtruth`

Then make sure to follow the next steps:

1. Remove the following files and folders, if present:
    - `./data/English_charDict.pkl`
    - `./data/HTMLTagDict.pkl`
    - `./data/nodesDetails/`
    - `./data/last.ckpt`
    - `./data/weights.ckpt`
2. Modify and execute the **generate.ipynb** notebook
    - Set the `vertical` and `attributes`, based on the content of `SWDE_Dataset/webpages/` and the description found in `SWDE_Dataset/readme.txt`
    - Set the number of friends to be used, the suggested values is `num_friends = 10`
    - Execute the notebook
3. Modify and execute the **train.ipynb** notebook
    - Set the `vertical` and `attributes`, based on the content of `SWDE_Dataset/webpages/` and the description found in `SWDE_Dataset/readme.txt`
    - Set the training, validation and testing website lists: `train_websites`, `val_websites`, and `test_websites`
    - Execute the notebook
        - The execution may fail while model training with:
        
        > torch.multiprocessing.spawn.ProcessExitedException: process 0 terminated with signal SIGSEGV` 
        
        - This indicates that it ran out of memory.
        - The solution is to reduce the training set by considering fewer web-sites.

## Getting more GloVe features

Pre-trained GloVe features and the Glove implementation is available from **[here](https://nlp.stanford.edu/projects/glove/)**.