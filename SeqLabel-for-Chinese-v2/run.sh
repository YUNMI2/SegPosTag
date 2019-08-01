rm -rf ./result/*
rm -rf ./save/*
#python3 train.py --gpu 1
nohup python3 train.py --gpu 1 --thread 4 > ./result/log.SegPos 2>&1 &
