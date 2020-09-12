source activate hv
script hv_train.log
CUDA_VISIBLE_DEVICES=0
python -u main.py --config=hv.yaml
exit
