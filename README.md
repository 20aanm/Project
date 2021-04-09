# Project

# Requirements
  * python 3.5
  * tensorflow 1.5
  
  
# Data
* Used the same data they have used for the graphtransformer https://github.com/sodawater/GraphTransformer.
# Train GraphTransformer
  * python TrainGT.py --enc_layers=8 --dec_layers=6 --num_heads=2 --num_units=256 --emb_dim=300  --train_dir=save/ --use_copy=1 --batch_size=16 --dropout_rate=0.2 --gpu_device=0 --max_src_len=90 --max_tgt_len=90
# Test GraphTransformer
  * python TestGT.py --enc_layers=8 --dec_layers=6 --num_heads=2 --num_units=256 --emb_dim=300  --train_dir=save/ --use_copy=1 --batch_size=16 --dropout_rate=0.2 --gpu_device=0 --max_src_len=90 --max_tgt_len=90

# Train DCGCN
* python Train(DCGCN).py --enc_layers=8 --dec_layers=6 --gcn_num_layers=4 --gcn_dropout=0.5 --gcn_num_hidden=480 --num_heads=2 --num_units=256 --emb_dim=300  --train_dir=save/ --use_copy=1 --batch_size=16 --dropout_rate=0.2 --gpu_device=0 --max_src_len=90 --max_tgt_len=90
# Test DCGCN
* python Test(DCGCN).py --enc_layers=8 --dec_layers=6 --gcn_num_layers=4 --gcn_dropout=0.5 --gcn_num_hidden=480 --num_heads=2 --num_units=256 --emb_dim=300  --train_dir=save/ --use_copy=1 --batch_size=16 --dropout_rate=0.2 --gpu_device=0 --max_src_len=90 --max_tgt_len=90

# Train LDGCN
* python Train(LDGCN).py --enc_layers=8 --dec_layers=6 --gcn_num_layers=4 --gcn_dropout=0.5 --gcn_num_hidden=480 --num_heads=2 --num_units=256 --emb_dim=300  --train_dir=save/ --use_copy=1 --batch_size=16 --dropout_rate=0.2 --gpu_device=0 --max_src_len=90 --max_tgt_len=90

# Test LDGCN
* python Test(LDGCN).py --enc_layers=8 --dec_layers=6 --gcn_num_layers=4 --gcn_dropout=0.5 --gcn_num_hidden=480 --num_heads=2 --num_units=256 --emb_dim=300  --train_dir=save/ --use_copy=1 --batch_size=16 --dropout_rate=0.2 --gpu_device=0 --max_src_len=90 --max_tgt_len=90

# Implemenation
* DCGCN
* ![image](https://user-images.githubusercontent.com/77679146/114119100-04640400-98b8-11eb-8312-df203d463d81.png)

