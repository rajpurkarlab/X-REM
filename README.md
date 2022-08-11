# CXR_ReFusE

## Preparation

CXR_ReFusE makes use of multiple github repos. Make sure to include all of them under `CXR_ReFusE` directory. 

* [ifcc](https://github.com/ysmiura/ifcc)
    * download model_medrad_19k.tar.gz by running `resources/download.sh`
    * Merge `ifcc_CXR_ReFusE_code` with the original directory
* [CheXbert](https://github.com/stanfordmlgroup/CheXbert)
    * download the [pretrained weights](https://stanfordmedicine.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9) 
* [CXR-RePaiR](https://github.com/rajpurkarlab/CXR-RePaiR)
* [CXR-Report-Metric](https://github.com/rajpurkarlab/CXR-Report-Metric)
   * Merge `CXR-Report-Metric_CXR_ReFusE_code` with the original directory

As we made multiple edits to the ALBEF directory, please refer to the ALBEF directory uploaded here instead of cloning a new one. 

Download our [pretrained checkpoints](https://drive.google.com/file/d/11UorBbh5cOcDfIzy_lCgMdn0zThvpDbp/view?usp=sharing) here!

## Data

Download the train/test reports and images from [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/). You may have to request for an approval to access the files.

## Environment

Create a conda environment for CXR_ReFusE

```
conda env create -f environment.yml -n cxr_refuse_env
```

Activate the environment

```
conda activate cxr_refuse_env
```

## Preprocessing
Refer to the data preprocessing step in [CXR-RePaiR](https://github.com/rajpurkarlab/CXR-RePaiR) to acquire `mimic_train_impressions.csv`, `mimic_test_impressions.csv`, and `cxr.h5`.  

Preprocess data to be used for pre-training ALBEF

```
preprocess_mimic.py --data_dir <path to MIMIC>  --impressions_train_path <path to mimic_train_impressions.csv> --impressions_test_path <path to mimic_test_impressions.csv> --out_dir <path to store the processed data>
```

## Training

Pretrain ALBEF
```
cd ALBEF 
sh pretrain_script.sh
```

Generate the train file for finetuning ALBEF on visual entailment task 
```
python generate_ve_train.py
```
Finetune the ALBEF model on visual entailment task 
```
cd ALBEF 
sh ve_script.sh
```

## Inference

```
sh inference.sh
```

`inference.sh` generates a dataframe containing two columns: "Report Impression" column holding reports before applying the nli filter, and 
"filtered" column containing the filtered reports. 
    
## Evaluation


```
cd CXR-Report-Metric
conda activate <environment name for CXR-Report-Metric>
python prepare_df.py --fpath <input path> --opath <output path>
python test_metric.py
python3 compute_avg_score.py --fpath <input path>
```
Refer to [CXR-Report-Metric](https://github.com/rajpurkarlab/CXR-Report-Metric) for a detailed explanation on the metric 

## Supplementary Experiments

Generate reports without using the visual entailment scores 
```
sh inference_no_ve.sh
```

Generate reports without the nli filter
```
sh inference_no_nli.sh
```

Replace the nli filter with bertscore as the metric for measuring redundancy
```
sh inference_bertscore.sh
```
