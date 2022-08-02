# CXR_ReFusE

![](https://i.imgur.com/J5Zfdke.png)

We propose [one sentence blurb goes here]. See our paper [here]()!

## Setup

CXR_ReFusE makes use of multiple GitHub repos. To set up the complete CXR_ReFusE directory, perform the following commands inside `CXR_ReFusE/`:

* [ifcc](https://github.com/ysmiura/ifcc)

```bash
git clone https://github.com/ysmiura/ifcc
cd ifcc
sh resources/download.sh
cd ..
mv ifcc_CXR_ReFusE_code/* ifcc/
rm -rf ifcc_CXR_ReFusE_code
```

* [CheXbert](https://github.com/stanfordmlgroup/CheXbert)

```bash
git clone https://github.com/stanfordmlgroup/CheXbert
cd CheXbert
mkdir models
cd models
```
> Once inside the `models` directory, download the [pretrained weights](https://stanfordmedicine.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9) for CheXbert.

* [CXR-RePaiR](https://github.com/rajpurkarlab/CXR-RePaiR)

```bash
git clone https://github.com/rajpurkarlab/CXR-RePaiR
```

* [CXR-Report-Metric](https://github.com/rajpurkarlab/CXR-Report-Metric)
  
```bash
git clone https://github.com/rajpurkarlab/CXR-Report-Metric
mv CXR-Report-Metric_CXR_ReFusE_code/* CXR-Report-Metric/
rm -rf CXR-Report-Metric_CXR_ReFusE_code
```

* [ALBEF](https://github.com/salesforce/ALBEF)

> As we made multiple edits to the ALBEF directory, please refer to the ALBEF directory uploaded here instead of cloning a new one. Make sure to download [ALBEF_4M.pth](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF_4M.pth) (`wget https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF_4M.pth`) and place it in `ALBEF/`.


## Data Preprocessing

Here, we make use of CXR-RePaiR's data preprocessing steps:

> #### Environment Setup
```bash
cd CXR-RePaiR
conda env create -f cxr-repair-env.yml
conda activate cxr-repair-env
```

>#### Data Access

>First, you must get approval for the use of [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) and [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/). With approval, you will have access to the train/test reports and the JPG images.

> #### Create Data Split
```bash
python data_preprocessing/split_mimic.py \
  --report_files_dir=<directory containing all reports> \
  --split_path=<path to split file in mimic-cxr-jpg> \
  --out_dir=mimic_data
```

> #### Extract Impressions Section
```bash
python data_preprocessing/extract_impressions.py \
  --dir=mimic_data
```

> #### Create Test Set of Report/CXR Pairs
```bash
python data_preprocessing/create_bootstrapped_testset_copy.py \
  --dir=mimic_data \
  --bootstrap_dir=bootstrap_test \
  --cxr_files_dir=<mimic-cxr-jpg directory containing chest X-rays>
```
<br>

The above commands produce the following files: `cxr.h5`, `mimic_train_impressions.csv`, and `mimic_test_impressions.csv`. 

To remove references to priors from the above data files, move the above three files to a new folder `CXR_ReFusE/data`, then run:

```bash
python remove_prior_refs.py
```

<br>

To skip these steps and instead obtain the final data, perform one of the following:

Download the `data` folder [here](https://drive.google.com/file/d/1gAncFcsW29bwQMDAajvxrMYPD5VehXTE/view?usp=sharing) and place it in the `CXR_ReFusE` directory.

--OR--

Run the following inside `CXR_ReFusE`:
```bash
pip install gdown
gdown 1gAncFcsW29bwQMDAajvxrMYPD5VehXTE
unzip data.zip
rm data.zip
```

## Training
To pretrain ALBEF, run:

```bash
cd ALBEF
sh pretrain_script.sh
```

Linked here are the ALBEF model checkpoints [with](https://www.dropbox.com/s/b4tkf2z4v6wa4zj/checkpoint_59.pth?dl=0) and [without](https://drive.google.com/file/d/183TClsB_fzCOHa6ESWfefV6EoN0EfmMI/view?usp=sharing) removing references to priors from the MIMIC-CXR reports corpus.

> TODO: Include fine-tuning script, if needed.
<!-- For finetuning, use `data/ve_train.json` that is generated by `data/generate_ve_train_file.py`.  -->

## Inference

To run inference, perform the following commands:

<!-- cd ifcc
conda env create -f environment.yml -n m2trans
conda activate m2trans
cd .. -->

```bash
sh inference.sh
```

This script first calls `ALBEF/CXR_ReFusE_pipeline.py` to use cosine-sim scores to select k' reports, then calls `ifcc/m2trans_nli_filter.py` to select k reports based on nli scores.

If using the visual entailment model, change the line `python3 CXR_ReFusE_pipeline.py` in `inference.sh` to `python3 CXR_ReFusE_pipeline.py --use_ve=True`.

> In the resulting CSV file, the "Report Impression" column contains the reports before applying the nli filter, and the "Filtered" column contains the filtered reports.
    
## Evaluation
For evaluating the generated reports, we make use of CXR-Report-Metric:

### Setup

```bash
cd CXR-Report-Metric
conda create -n "myenv" python=3.7.0 ipython
conda activate myenv
pip install -r requirements.txt
```

Next, download the RadGraph model checkpoint from PhysioNet [here](https://physionet.org/content/radgraph/1.0.0/). The checkpoint file can be found under the "Files" section at path `models/model_checkpoint/`. Set `RADGRAPH_PATH` in `config.py` to the path to the downloaded checkpoint.

### Evaluation

1. Use `prepare_df.py` to select the inferences for the corresponding 2,192 samples from our generation. (If using the visual entailment model, make sure to run `python prepare_df.py --use_ve=True`).

2. In `config.py`, set `GT_REPORTS` to `../data/cxr2_generated_appa.csv` and `PREDICTED_REPORTS` to `pred.csv`. Set `OUT_FILE` to the desired path for the output metric scores. Set `CHEXBERT_PATH` to the path to the downloaded checkpoint (`CheXbert/models/chexbert.pth`).

3. Use `test_metric.py` to generate the scores. 

4. Finally, use `compute_avg_score.py` to output the average scores. 

<br>

## Download pre-trained BioBERT models

The above steps to run CXR_ReFusE rely on two pretrained models:

### GILBERT: <u>G</u>enerating <u>I</u>n-text <u>L</u>abels of References to Priors with Bio<u>BERT</u>
<!-- Remove token-level references to priors -->
The BioBERT model for token tagging (marking each token as "REMOVE" or "KEEP," based on whether they comprise a reference to a prior) trained on the [annotated MIMIC-CXR train set]() is available [here](https://huggingface.co/rajpurkarlab/gilbert). 

> TODO: Upload [modified data](https://drive.google.com/file/d/1Pepjgl96_m3HfMUPpDoVeu32US4MXs6T/view?usp=sharing) to PhysioNet, and link this in the above paragraph.

The model can be called programmatically as follows:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

def get_pipe():
    model_name = "rajpurkarlab/gilbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    pipe = pipeline(task="token-classification", model=model.to("cpu"), tokenizer=tokenizer, aggregation_strategy="simple")
    return pipe

def remove_priors(pipe, report):
    ret = ""
    for sentence in report.split("."):
        if sentence and not sentence.isspace():
            p = pipe(sentence)
            string = ""
            for item in p:
                if item['entity_group'] == 'KEEP':
                    string += item['word'] + " "
            ret += string.strip().replace("redemonstrate", "demonstrate").capitalize() + ". "
    return ret.strip()

modified_report = remove_priors(get_pipe(), "YOUR REPORT")
```


### FilBERT: <u>Fil</u>tering Sentence-Level References to Priors with Bio<u>BERT</u>
The BioBERT model used to classify sentences as containing references to priors or not (marking each sentence as either label = 1 or label = 0, where 1 and 0 correspond to containing prior references and containing no references, respectively) trained on the [annotated MIMIC-CXR train set]() is available [here](https://huggingface.co/rajpurkarlab/filbert). 

The model can be called programmatically as follows:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_name="rajpurkarlab/filbert"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def run_bert_classifier(sentence, tokenizer, model):
    pipe = pipeline("sentiment-analysis", model=model.to("cpu"), tokenizer=tokenizer)
    return int(pipe(sentence)[0]['label'][-1])

tokenizer, model = load_model()
run_bert_classifier("SINGLE SENTENCE FROM A REPORT", tokenizer, model)
```


## Retraining GILBERT and FilBERT
To retrain GILBERT and FilBERT, see [here]().

> TODO: Link these [train](https://colab.research.google.com/drive/12sM1baPnoTAEYkumzglM-UkIqgbZzZ3q?usp=sharing) [scripts](https://colab.research.google.com/drive/1UsFRNyldb_BbPfpk_CQR2Xs3o1r2V2G-), once they are uploaded to GitHub.

<br>

## Citing
If you are making use of this repository, please cite the following paper:

```
 
```
