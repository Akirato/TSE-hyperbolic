
## Multi-relational Taylor Series Hyperbolic Model

Code based on [MuRP](https://github.com/ibalazevic/multirelational-poincare)
### Running a model

To run the model, execute the following command:

    CUDA_VISIBLE_DEVICES=0 python main.py --model taylor --dataset WN18RR --num_iterations 500 
                                           --nneg 50 --batch_size 128 --lr 50 --dim 40 

Available datasets are:
    
    FB15k-237
    WN18RR
    
To reproduce the results from the paper, use learning rate 50 for WN18RR and learning rate 10 for FB15k-237.


### Requirements

The codebase is implemented in Python 3.6.6. Required packages are:

    numpy      1.15.1
    pytorch    1.0.1
    


