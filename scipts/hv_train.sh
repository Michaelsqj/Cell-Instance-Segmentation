# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/shenqijia/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/shenqijia/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/shenqijia/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/shenqijia/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate hv
CUDA_VISIBLE_DEVICES=0 python -u main.py --config=hv.yaml >hv_train.log
