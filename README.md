# LSDRec

This is our paper's PyTorch implementation, **Debiased Sequential Recommendation by Separating Long-Term and Short-Term Interests**，built on the RecBole library.

* The model implementation is at `recbole/model/sequential_recommender/lsdrec.py`


## Environments
* Python 3.7
* torch>=1.10.0

## Datasets
* Reddit: A resource for studying users' interactions about topics on Reddit. 
* Beauty: Collected from the Amazon Beauty product category, detailing user interactions with these products. 
* Sports: Gathering user product interaction data from the Amazon Sports product category. 
* ML-1M: Containing users' rating behavior for movies.

## Run the codes

On Reddit dataset:

`python main.py --dataset=reddit `
