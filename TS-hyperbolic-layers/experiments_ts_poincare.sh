export HGCN_HOME=$(pwd)
export LOG_DIR="$HGCN_HOME/logs"
export PYTHONPATH="$HGCN_HOME:$PYTHONPATH"
export DATAPATH="$HGCN_HOME/data"
CUDA_VISIBLE_DEVICES=0 python train.py --task lp --dataset citeseer --model HAT --lr 0.1 --dim 16 --num-layers 3 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold TSPoincareBall --log-freq 5 --cuda 0 --c None
