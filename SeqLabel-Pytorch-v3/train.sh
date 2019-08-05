rm -rf ./result/*
rm -rf ./save/*
nohup python3 $(pwd)/train.py --gpu 5 --thread 4 > ./result/log.SegPos.train 2>&1 &
