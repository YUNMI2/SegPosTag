# SEG POS Tagging 
```
Model1 : BiLSTM + CRF
Model2 : BiLSTM + SoftMax
```

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

## update
```
2019-09-05 15:36  # fix a bug in computing softmax loss of dev
2019-09-04 15:20  # fix some bugs in compute prf value and rewrite the predict format
2019-08-20 15:40  # fix some bugs in Test
2019-08-14 17:23  # fix some bugs in Test, now can assign gpu id
2019-08-13 18:01  # rewrite train&&test&&config, add Pytorch Distributed Training Mode, now support four mode[SingleCPU/SingleGPU/MultiGPU/DistGPU]
2019-08-08 19:45  # fix some bugs in softmax loss
2019-08-08 17:01  # besides crf-loss, add softmax loss
2019-08-08 11:45  # add functions of assigning gpu id and saveing&&loading model on multiGPU version
2019-08-07 20:33  # fix some bugs on multiGPU version
2019-08-07 16:39  # rewrite train, now train support multiGPU, but test is on the way
2019-08-06 17:00  # split train data and need less memory 
```
