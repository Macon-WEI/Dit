torchrun --nnodes=1 \
--nproc_per_node=1 \
./train_options/train_inpaint.py \
--model DiT-L/4 \
--train-data-path /home/tongji209/majiawei/Dit/dataset/train/source-10 \
--gt-data-path /home/tongji209/majiawei/Dit/dataset/train/target-10 \
--results-dir inpaint-train-results \
--log-every 1 \
--global-batch-size 10 \
--epochs 600
