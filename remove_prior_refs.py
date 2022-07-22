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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove prior references from report impressions')
    parser.add_argument('--dir', type=str, default='data/', help='directory where impression sections are stored')
    args = parser.parse_args()
    
    pipe = get_pipe()

    #args.dir
