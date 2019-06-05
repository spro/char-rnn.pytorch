# char-rnn.pytorch

A PyTorch implementation of [char-rnn](https://github.com/karpathy/char-rnn) for character-level text generation. This is copied from [the Practical PyTorch series](https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb).

## Training

Download [this Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (from the original char-rnn) as `shakespeare.txt`.  Or bring your own dataset &mdash; it should be a plain text file (preferably ASCII).

Run `train.py` with the dataset filename to train and save the network:

```
> python train.py --train shakespeare.txt

Training for 2000 epochs...
(... 10 minutes later ...)
Saved as shakespeare.pt
```
After training the model will be saved as `[filename].pt`.
According to the --print_every arg model checkpoints will be saved in the `Save/` folder that should be in the same fold where the train.py script is called

### Training options

```
Usage: train.py [options]

Options:
--train            Train data
--valid            Validation data
--model            Whether to use LSTM or GRU units         gru
--n_epochs         Number of epochs to train                10
--print_every      Log learning rate at this interval       100
--hidden_size      Hidden size of GRU                       50
--n_layers         Number of GRU layers                     2
--learning_rate    Learning rate                            0.01
--chunk_len        Length of training chunks                200
--batch_size       Number of examples per batch             100
--batch_type       Batch random (0) or sequential (1)       0
--drop_out         drop-out rate between Recurrent layers   0
--early_stopping   Number of validation step with no impr.  10
--model_name       model(session) name, used in checkpoints 

--cuda             Use CUDA
```

## Generation

Run `generate.py` with the saved model from training, and a "priming string" to start the text with.

```
> python generate.py shakespeare.pt --prime_str "Where"

Where, you, and if to our with his drid's
Weasteria nobrand this by then.

AUTENES:
It his zersit at he
```

### Generation options
```
Usage: generate.py [filename] [options]

Options:
-p, --prime_str      String to prime generation with
-l, --predict_len    Length of prediction
-t, --temperature    Temperature (higher is more chaotic)
--cuda               Use CUDA
```


### Grid search
```
Usage : search_params.py --train [options]

Hard-coded params :
  -learning_rate : [0.001,0.01]
  -max_epochs : [500]
  -n_batch_size : [32,1024] (should be changed according to available memory)
  -batch_type : [0,1] (random vs consequent sampling)
  -model_type : [lstm, gru]
  
Options:
--train     training file
--valid     validation file
--hidden_size_init    50
--hidden_size_end     800
--hidden_size_step    200
--n_layer_init        1
--n_layer_end         4
--n_layer_step        1
--chunk_len_init      20
--chunk_len_end       90
--chunk_len_step      10
--early_stopping      10
--optimizer           adam 
--cuda                
```

### TODO
[x] Grid search
  -[] Adding dropout to grid search (config. with less than 1 layer doesn't need dropout!!)
  -[] Adapt batch_size to available memory
[x] Early stopping
[x] Add Dropout (p)
[x] Add Validation set to monitor overfitting
[x] Saving model at checkpoint
[x] Saving train and validation error, with training params to file
[x] Refact to more OO paradigm

```

