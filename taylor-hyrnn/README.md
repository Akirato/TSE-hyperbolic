Hyperbolic RNN in Pytorch
=========================

## Prerequisites
You need `Python3.6` to run the code. The list of the dependencies are in `requrements.txt`. Run:
```
python -m pip install -r requirements.txt
```

## Training

To reproduce the results from the paper, run the following command:
The hyperbolic gru and rnn are both of the Talyor series formulation.
To get the original hyperbolic formulation, refer https://github.com/ferrine/hyrnn

```
cd ./experiment_run
python run.py --data_dir=./data --num_epochs=30 --log_dir=./logs --batch_size=1024 --num_layers=2 --cell_type=hyp_gru
```

where you can change the argument `cell_type=hyp_gru` to `cell_type=eucl_gru` if you want to run Euclidean version of GRU.
