# CXR_ReFusE

## Setup

CXR_ReFusE makes use of multiple github repos. Make sure to include all of them under CXR_ReFusE directory. 

* [ifcc](https://github.com/ysmiura/ifcc)
    * download model_medrad_19k.tar.gz by running `resources/download.sh`
    * Merge `ifcc_CXR_ReFusE_code` with the original directory
* [CheXbert](https://github.com/stanfordmlgroup/CheXbert)
    * download the [pretrained weights](https://stanfordmedicine.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9) 
* [CXR-RePaiR](https://github.com/rajpurkarlab/CXR-RePaiR)
* [CXR-Report-Metric](https://github.com/rajpurkarlab/CXR-Report-Metric)
   * Merge `CXR-Report-Metric_CXR_ReFusE_code` with the original directory

As we made multiple edits to the ALBEF directory, please refer to the ALBEF directory uploaded here instead of cloning a new one. 

Download the [zipfile](https://drive.google.com/file/d/1VW8q0b4Jh6Pj3crpFHTRC3mTCRUjI2zi/view?usp=sharing) containing our dataset and place them in the appropriate folders. 


## Data-preprocessing
Here we use CXR-RePaiR's data preprocessing steps.
Refer to "Data Preprocessing" in cxr-repair for more detail. 
We obtain `data/cxr.h5`, `data/mimic_train_impressions.csv`, `data/mimic_test_impressions.csv` from CXR-RePaiR

## Training

To pretrain the ALBEF model, run 
```
cd ALBEF 
sh pretrain_script.sh
```

To generate the train file for finetuning ALBEF on visual entailment task, run 
```
python generate_ve_train.py
```
To finetune the ALBEF model on visual entailment task, run 
```
cd ALBEF 
sh ve_script.sh
```

## Inference

To generate reports, run 
```
sh inference.sh
```

`inference.sh` generates a dataframe containing two columns: "Report Impression" column holding reports before applying the nli filter, and 
"filtered" column containing the filtered reports. 
    
## Evaluation

For evaluating the generated reports, we use [CXR-Report-Metric](https://github.com/rajpurkarlab/CXR-Report-Metric)

```
cd CXR-Report-Metric
conda activate <environment name for CXR-Report-Metric>
python prepare_df.py --fpath <input path> --opath <output path>
python test_metric.py
python3 compute_avg_score.py --fpath <input path>
```

## Supplementary Experiments

To generate reports without using the visual entailment scores, run 
```
sh inference_no_ve.sh
```

To generate reports without the nli filter, run 
```
sh inference_no_nli.sh
```

To replace the nli filter with bertscore as the metric for measuring redundancy, run 
```
sh inference_bertscore.sh
```
