#nohup python3 $(pwd)/train.py --gpu 0,1,2,3 --thread 4 > ./result/log.SegPos.train.noshuffle.256 2>&1 &
#nohup python3 $(pwd)/train.py --gpu 0 --thread 4 > ./result/log.SegPos.train 2>&1 &
nohup python3 $(pwd)/train.py --gpu 5,6 --thread 4 > ./result/log.SegPos.train.CTB5-test 2>&1 &
