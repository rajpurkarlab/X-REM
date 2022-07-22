import json
import csv
from tempfile import NamedTemporaryFile
import shutil
import argparse
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

modelname = "rajpurkarlab/biobert-finetuned-prior-rmv"
tokenizer = AutoTokenizer.from_pretrained(modelname)
model = AutoModelForTokenClassification.from_pretrained(modelname)

def get_pipe():
    model_name = "rajpurkarlab/biobert-finetuned-prior-rmv"
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

def update_json(pipe, path):
    f = open(path, "r")
    json_obj = json.load(f)
    f.close()
    for i in tqdm(range(len(json_obj))):
        item = json_obj[i]
        report = item['caption']
        item['caption'] = remove_priors(pipe, report)
    f = open(path, "w")
    json.dump(json_obj, f)
    f.close()

def update_csv(pipe, path):
    tempfile = NamedTemporaryFile(mode='w', delete=True)
    fields = ['Unnamed: 0', 'dicom_id', 'study_id', 'subject_id', 'report']

    with open(path, 'r') as csvfile, tempfile:
        reader = csv.DictReader(csvfile, fieldnames=fields)
        writer = csv.DictWriter(tempfile, fieldnames=fields)
        for row in reader:
            row = {'Unnamed: 0': row['Unnamed: 0'], 'dicom_id': row['dicom_id'], 'study_id': row['study_id'], 'subject_id': row['subject_id'], 'report': remove_priors(pipe, row['report'])}
            writer.writerow(row)
    
    shutil.move(tempfile.name, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove prior references from report impressions')
    parser.add_argument('--dir', type=str, default='data/', help='directory where impression sections are stored')
    args = parser.parse_args()
    
    pipe = get_pipe()

    train_json = args.dir + "mimic_train.json"
    train_csv = args.dir + "mimic_train_impressions.csv"
    test_csv = args.dir + "mimic_test_impressions.csv"

    update_json(pipe, train_json)
    
    for csv_path in [train_csv, test_csv]:
        update_csv(pipe, csv_path)
