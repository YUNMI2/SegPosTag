rm -rf ./result/*
rm -rf ./save/*
nohup python3 train.py --gpu 1 --thread 4 >> ./result/log.SegPos 2>&1 &
