# Project

# Requirements
  # python 3.5
  # tensorflow 1.5
  
# Train GraphTransformer

# Test GraphTransformer
  python Test(LDGCN).py --enc_layers=8 --dec_layers=6 --num_heads=2 --num_units=256 --emb_dim=300  --train_dir=save/ --use_copy=1 --batch_size=16 --dropout_rate=0.2 --gpu_device=0 --max_src_len=90 --max_tgt_len=90

# Train Densly connected 
