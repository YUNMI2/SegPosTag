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

## update log
```
2019-08-07 20:33  # fix some bugs on multiGPU version
2019-08-07 16:39  # rewrite train, now train support multiGPU, but test is on the way
2019-08-06 17:00  # split train data and need less memory 
```
