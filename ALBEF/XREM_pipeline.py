import pandas as pd
from XREM_module import RETRIEVAL_MODULE
import os
from tqdm import tqdm
import argparse

def main(args):
    df = pd.read_csv(args.impressions_path)
    impressions = df["report"].drop_duplicates().dropna().reset_index(drop = True)
    cosine_sim_module = RETRIEVAL_MODULE(impressions=impressions, 
                                         mode='cosine-sim', 
                                         config=args.albef_retrieval_config, 
                                         checkpoint=args.albef_retrieval_ckpt, 
                                         topk=args.albef_retrieval_top_k,
                                         input_resolution=256, 
                                         img_path=args.img_path, 
                                         delimiter=args.albef_retrieval_delimiter, 
                                         max_token_len = 25)
    output = cosine_sim_module.predict()
    
    if args.albef_itm_top_k > 0:
        new_impressions = [report.split(args.albef_retrieval_delimiter) for report in output['Report Impression']]
        itm_module = RETRIEVAL_MODULE(impressions=new_impressions, 
                                                mode='image-text-matching', 
                                                config=args.albef_itm_config, 
                                                checkpoint=args.albef_itm_ckpt,
                                                topk=args.albef_itm_top_k,
                                                input_resolution=384, 
                                                img_path=args.img_path, 
                                                delimiter=args.albef_itm_delimiter, 
                                                max_token_len=30)
        output = itm_module.predict()
    output.to_csv(args.save_path, index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--impressions_path', 
                        default='../data/mimic_train_impressions.csv', 
                        help='path to the mimic train corpus')
    parser.add_argument('--img_path', 
                        default='../data/cxr.h5', 
                        help='path to the test file') 
    parser.add_argument('--save_path', 
                        default='example.csv', 
                        help='path to store the output')
    parser.add_argument('--albef_retrieval_config', 
                        default='configs/Cosine-Retrieval.yaml', 
                        help='config file for the pre-trained albef model used for cosine similarity retrieval')
    parser.add_argument('--albef_retrieval_ckpt', 
                        default='output/sample/pretrain/checkpoint_59.pth', 
                        help='weights for the pre-trained albef model')
    parser.add_argument('--albef_retrieval_top_k', 
                        type=int, 
                        default=50, 
                        help='number of reports to retrieve at the cosine similarity retrieval step')
    parser.add_argument('--albef_retrieval_delimiter', 
                        default='[SEP]', 
                        help='delimiter used for the cosine similarity retrieval step')
    parser.add_argument('--albef_itm_config', 
                        default='configs/ITM.yaml', 
                        help='config file for the albef model fine-tuned on image-text matching (binary visual-entailment)')
    parser.add_argument('--albef_itm_ckpt', 
                        default='output/sample/itm/checkpoint_7.pth', 
                        help='weights for the fine-tuned albef model')    
    parser.add_argument('--albef_itm_top_k', 
                        default=10, 
                        type = int, 
                        help='number of reports to retrieve at the image-text matching retrieval step')
    parser.add_argument('--albef_itm_delimiter', 
                        default='[SEP]',
                        help='delimiter used for the image-text matching retrieval step')
    args = parser.parse_args()
    main(args)
