torchrun --nnodes=1 \
--nproc_per_node=1 \
./train_options/train_inpaint.py \
--model DiT-L/4 \
--train-data-path /home/tongji209/majiawei/Dit/dataset/train/source-1 \
--gt-data-path /home/tongji209/majiawei/Dit/dataset/train/target-1 \
--results-dir inpaint-train-results \
--log-every 1 \
--global-batch-size 1 \
--learning-rate 1e-5 \
--epochs 700
