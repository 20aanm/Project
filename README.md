# Project

# Requirements
  * python 3.5
  * tensorflow 1.5
  
  
# Data
* Used the same data they have used for the graphtransformer https://github.com/sodawater/GraphTransformer. So they uploaded some example format of the data they have used not the whole data which is used for the experiments.
# Train GraphTransformer
  * python TrainGT.py --enc_layers=8 --dec_layers=6 --num_heads=2 --num_units=256 --emb_dim=300  --train_dir=save/ --use_copy=1 --batch_size=16 --dropout_rate=0.2 --gpu_device=0 --max_src_len=90 --max_tgt_len=90
# Test GraphTransformer
```
python TestGT.py --enc_layers=8 --dec_layers=6 --num_heads=2 --num_units=256 --emb_dim=300  --train_dir=save/ --use_copy=1 --batch_size=16 --dropout_rate=0.2 --gpu_device=0 --max_src_len=90 --max_tgt_len=90
```
# Train DCGCN
```
python Train(DCGCN).py --enc_layers=8 --dec_layers=6 --gcn_num_layers=4 --gcn_dropout=0.5 --gcn_num_hidden=480 --num_heads=2 --num_units=256 --emb_dim=300  --train_dir=save/ --use_copy=1 --batch_size=16 --dropout_rate=0.2 --gpu_device=0 --max_src_len=90 --max_tgt_len=90
```
# Test DCGCN
```
python Test(DCGCN).py --enc_layers=8 --dec_layers=6 --gcn_num_layers=4 --gcn_dropout=0.5 --gcn_num_hidden=480 --num_heads=2 --num_units=256 --emb_dim=300  --train_dir=save/ --use_copy=1 --batch_size=16 --dropout_rate=0.2 --gpu_device=0 --max_src_len=90 --max_tgt_len=90
```
# Train LDGCN
```
python Train(LDGCN).py --enc_layers=8 --dec_layers=6 --gcn_num_layers=4 --gcn_dropout=0.5 --gcn_num_hidden=480 --num_heads=2 --num_units=256 --emb_dim=300  --train_dir=save/ --use_copy=1 --batch_size=16 --dropout_rate=0.2 --gpu_device=0 --max_src_len=90 --max_tgt_len=90
```
# Test LDGCN
```
python Test(LDGCN).py --enc_layers=8 --dec_layers=6 --gcn_num_layers=4 --gcn_dropout=0.5 --gcn_num_hidden=480 --num_heads=2 --num_units=256 --emb_dim=300  --train_dir=save/ --use_copy=1 --batch_size=16 --dropout_rate=0.2 --gpu_device=0 --max_src_len=90 --max_tgt_len=90
```
# Implemenation

* DCGCN

     ![image](https://user-images.githubusercontent.com/77679146/114119100-04640400-98b8-11eb-8312-df203d463d81.png)

* LDGCN

     ![image](https://user-images.githubusercontent.com/77679146/114119206-34aba280-98b8-11eb-9b41-3e2a39a56901.png)
     ![image](https://user-images.githubusercontent.com/77679146/114119220-3bd2b080-98b8-11eb-9a4e-5ad98c285112.png)


# Evaluation and Resutls
* For model evaluation, the resutls of expermiemnts are in the "results folder".
* The results folder contains the log file of the expermients alongside with the reference and prediction file for each experiment.


