# SEG POS Tagging 

Model : LSTM + CRF


## requirements

```
python >= 3.6.3
pytorch = 0.4.1
```

## config
```
config.seg = Ture     # do Seg or SegPos , F1 value evaluate
config.seg = False    # only do Pos, accuracy evaluate 
```

## running

```
./clean.sh  # remove all tmp file
./train.sh  # train a model 
./test.sh   # test a model 
```

