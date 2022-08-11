source ~/.bashrc

module load conda2/4.2.13
module load gcc/6.2.0
module load cuda/11.2

cd ALBEF

python3 CXR_ReFusE_pipeline.py --albef_retrieval_delimiter ' ' --save_path no_ve.csv --albef_retrieval_top_k 2 --albef_ve_top_k 0