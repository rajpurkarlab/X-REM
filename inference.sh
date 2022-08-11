source ~/.bashrc

module load conda2/4.2.13
module load gcc/6.2.0
module load cuda/11.2

cd ALBEF

python3 CXR_ReFusE_pipeline.py --save_path before_nli.csv

cd ../ifcc

conda activate m2trans

python3 m2trans_nli_filter.py --input_path before_nli.csv --save_path after_nli.csv

conda deactivate