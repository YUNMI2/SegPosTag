# Debug: Test Code 
#nohup python $(pwd)/test.py --config ./conf/bilstm.conf --gpu 0 --thread 4 --eval_file /search/odin/zhuyun/Data/WordSeg/CTB5/SEGPOS/origin/ctb5-test.segpos.conll --predict_file ./predict/test > ./result/log.test 2>&1 &


# Test Post Eng Data
#nohup python $(pwd)/test.py --config ./conf/google.bilstm.conf --gpu 3 --thread 4 --eval_file /search/odin/zhuyun/Data/Sogou/PostData/convert/log.eng.conll --predict_file ./predict/eng.googlebilstm.conll > ./result/log.test.eng 2>&1 &

# Test Post Eng2 Data
#nohup python $(pwd)/test.py --config ./conf/google.bilstm.conf --gpu 3 --thread 4 --eval_file /search/odin/zhuyun/Data/Sogou/PostData/convert/log.eng2.conll --predict_file ./predict/eng2.googlebilstm.conll > ./result/log.test.eng2 2>&1 &

# Test Post Eng Data add Punct
#nohup python $(pwd)/test.py --config ./conf/google.bilstm.conf --gpu 3 --thread 4 --eval_file /search/odin/zhuyun/Data/Sogou/PostData/convert/log.eng.addPunct.conll --predict_file ./predict/eng.addPunct.googlebilstm.conll > ./result/log.test.eng.addPunct 2>&1 &

# Test Post Eng2 Data
#nohup python $(pwd)/test.py --config ./conf/google.bilstm.conf --gpu 3 --thread 4 --eval_file /search/odin/zhuyun/Data/Sogou/PostData/convert/log.eng2.addPunct.conll --predict_file ./predict/eng2.addPunct.googlebilstm.conll > ./result/log.test.eng2.addPunct 2>&1 &

# Test Post enOnline Data
nohup python $(pwd)/test.py --config ./conf/google.bilstm.conf --gpu 4 --thread 4 --eval_file /search/odin/zhuyun/Data/Sogou/PostData/convert/log.en-online.conll --predict_file ./predict/en-online.googlebilstm.conll > ./result/log.test.en-online 2>&1 &

# Test Post enOnline Data and add Punct
nohup python $(pwd)/test.py --config ./conf/google.bilstm.conf --gpu 5 --thread 4 --eval_file /search/odin/zhuyun/Data/Sogou/PostData/convert/log.en-online.addPunct.conll --predict_file ./predict/en-online.addPunct.googlebilstm.conll > ./result/log.test.en-online.addPunct 2>&1 &
