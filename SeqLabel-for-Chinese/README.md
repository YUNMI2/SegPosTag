# SEG POS Tagging 

Model : LSTM + CRF


## requirements

```
python >= 3.6.3
pytorch = 0.4.1
```

## running

```
mkdir save                   # or define other path to save model and vocab
python train.py --pre_emb    # train the model with pretrained embedding
python evaluate.py           # reload the model and evaluate test file
```

## results

```
dev=95.84%
test=95.65%
```

