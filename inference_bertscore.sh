source ~/.bashrc

module load conda2/4.2.13
module load gcc/6.2.0
module load cuda/11.2

cd ALBEF

python3 CXR_ReFusE_pipeline.py --save_path before_bertscore.csv

python3 bertscore_filter.py --input_path before_bertscore.csv --save_path after_bertscore.csv
