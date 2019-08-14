# SingleCPU
#nohup python3 $(pwd)/train.py --data CTB5-POS --gpu SingleCPU --model bilstm  > ./result/log.SEGPOS.SingleCPU 2>&1 & 

# SingleGPU
#nohup python3 $(pwd)/train.py --data CTB5-POS --gpu SingleGPU --model bilstm  > ./result/log.SEGPOS.SingleGPU 2>&1 & 

# MultiGPU
#nohup python3 $(pwd)/train.py --data CTB5-POS --gpu MultiGPU --model bilstm  > ./result/log.SEGPOS.MultiGPU 2>&1 & 

# DistGPU
nohup python3 $(pwd)/train.py --data Google --gpu DistGPU --model bilstm --rank 0 > ./result/log.google.DistGPUNode1 2>&1 & 
nohup python3 $(pwd)/train.py --data Google --gpu DistGPU --model bilstm --rank 1 > ./result/log.google.DistGPUNode2 2>&1 & 
nohup python3 $(pwd)/train.py --data Google --gpu DistGPU --model bilstm --rank 2 > ./result/log.google.DistGPUNode3 2>&1 & 
nohup python3 $(pwd)/train.py --data Google --gpu DistGPU --model bilstm --rank 3 > ./result/log.google.DistGPUNode4 2>&1 & 
