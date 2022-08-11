import pandas as pd
from CXR_ReFusE_module import RETRIEVAL_MODULE
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
    
    if args.albef_ve_top_k > 0:
        new_impressions = [el.split(args.albef_retrieval_delimiter) for el in output['Report Impression']]
        ve_module = RETRIEVAL_MODULE(impressions=new_impressions, 
                                                mode='visual-entailment', 
                                                config=args.albef_ve_config, 
                                                checkpoint=args.albef_ve_ckpt,
                                                topk=args.albef_ve_top_k,
                                                input_resolution=384, 
                                                img_path=args.img_path, 
                                                delimiter=args.albef_ve_delimiter, 
                                                max_token_len=30)
        output = ve_module.predict()
    output.to_csv(args.save_path, index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--impressions_path', default='../data/mimic_train_impressions.csv', help= 'path to the mimic train corpus')
    parser.add_argument('--img_path', default='../data/cxr.h5', help = 'path to the test file') 
    parser.add_argument('--save_path', default='example.csv')
    parser.add_argument('--albef_retrieval_config', default='configs/Retrieval_flickr.yaml')
    parser.add_argument('--albef_retrieval_ckpt', default='output/sample/pretrain/checkpoint_59.pth')
    parser.add_argument('--albef_retrieval_top_k', type = int, default=50)
    parser.add_argument('--albef_retrieval_delimiter', default='[SEP]')
    parser.add_argument('--albef_ve_config', default='configs/VE.yaml')
    parser.add_argument('--albef_ve_ckpt', default='output/sample/ve/checkpoint_7.pth')    
    parser.add_argument('--albef_ve_top_k', default=10, type = int, help='url used to set up distributed training')
    parser.add_argument('--albef_ve_delimiter', default='[SEP]')
    args = parser.parse_args()
    main(args)
