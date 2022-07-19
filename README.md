# CXR_ReFusE

## Setup

CXR_ReFusE makes use of multiple github repos. Make sure to include all of them under CXR_ReFusE directory. 

* [ifcc](https://github.com/ysmiura/ifcc)
    * download model_medrad_19k.tar.gz by running resources/download.sh
* [CheXbert](https://github.com/stanfordmlgroup/CheXbert)
    * download pretrained weights
* [CXR-RePaiR](https://github.com/rajpurkarlab/CXR-RePaiR)
* [CXR-Report-Metric](https://github.com/rajpurkarlab/CXR-Report-Metric)

As we made multiple edits to the ALBEF directory, please refer to the ALBEF directory uploaded here instead of cloning a new one. 

Download the zipfile containing the dataset and place them in the appropriate folders. 


## Data-preprocessing
Use cxr-repair's data preprocessing steps.
Refer to "Data Preprocessing" in cxr-repair for more detail. 
We obtain `data/cxr.h5`, `data/mimic_train_impressions.csv`, `data/mimic_test_impressions.csv`

## Training

For pretraining & fine-tuning ALBEF, use `pretrain_script.sh` and ve_script.sh in `ALBEF` directory. 
For pretraining, use `data/mimic_train.json`
For finetuning, use `data/ve_train.json` (generated by `generate_ve_train_file.py`)
`Jaehwan_edits.txt` keeps track of the edits I made to the original ALBEF github codebase. 

## Inference
`inference.sh` first calls `ALBEF/CXR_ReFusE_pipeline.py` to use cosine-sim & ve scores to select k' reports. 
`inference.sh` then calls `ifcc/m2trans_nli_filter.py` to select k reports based on nli scores
"Report Impression" column contains the reports before applying the nli filter, 
"filtered" column contains the filtered reports. 
    
## Evaluation
For evaluating the generated reports, use CXR-Report-Metric. 
Note that the original paper computed the metric scores using a subset of the mimic test set. 
First activate the conda env for CXR-Report-Metric
Next use `prepare_df.py` to select the inferences for the corresponding 2,192 samples from our generation. 
Then use `test_metric.py` to generate the scores. 
Finally use `compute_avg_score.py` to find the average scores. 
