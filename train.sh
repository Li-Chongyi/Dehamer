# CUDA_VISIBLE_DEVICES=0,1 python src/train.py \
#   --dataset-name NH \
#   --train-dir ./data/train_NH/ \
#   --valid-dir ./data/valid_NH/ \
#   --ckpt-save-path ../ckpts \
#   --ckpt-overwrite \
#   --nb-epochs 5000 \
#   --batch-size 2\
#   --train-size 800 1200  \
#   --valid-size 800 1200 \
#   --loss l1 \
#   --plot-stats \
#   --cuda   

# CUDA_VISIBLE_DEVICES=1,2 python src/train.py \
#   --dataset-name dense \
#   --train-dir ./data/train_dense/ \
#   --valid-dir ./data/valid_dense/ \
#   --ckpt-save-path ../ckpts \
#   --ckpt-overwrite \
#   --nb-epochs 5000 \
#   --batch-size 2\
#   --train-size 800 1200  \
#   --valid-size 800 1200 \
#   --loss l1 \
#   --plot-stats \
#   --cuda   

CUDA_VISIBLE_DEVICES=1,2 python src/train.py \
  --dataset-name indoor \
  --train-dir ./data/train_indoor/ \
  --valid-dir ./data/valid_indoor/ \
  --ckpt-save-path ../ckpts \
  --ckpt-overwrite \
  --nb-epochs 5000 \
  --batch-size 2\
  --train-size 512 512  \
  --valid-size 512 512 \
  --loss l1 \
  --plot-stats \
  --cuda   

# CUDA_VISIBLE_DEVICES=1,2 python src/train.py \
#   --dataset-name outdoor \
#   --train-dir ./data/train_outdoor/ \
#   --valid-dir ./data/valid_outdoor/ \
#   --ckpt-save-path ../ckpts \
#   --ckpt-overwrite \
#   --nb-epochs 5000 \
#   --batch-size 2\
#   --train-size 512 512  \
#   --valid-size 512 512 \
#   --loss l1 \
#   --plot-stats \
#   --cuda   