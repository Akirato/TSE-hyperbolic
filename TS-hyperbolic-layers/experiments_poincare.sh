export HGCN_HOME=$(pwd)
export LOG_DIR="$HGCN_HOME/logs"
export PYTHONPATH="$HGCN_HOME:$PYTHONPATH"
export DATAPATH="$HGCN_HOME/data"
CUDA_VISIBLE_DEVICES=0,1 python train.py --task nc --dataset cora --model HAT --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall --log-freq 5 --cuda 0 --c None
