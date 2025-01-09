torchrun --nnodes=1 \
--nproc_per_node=1 \
./train_options/train_inpaint.py \
--model DiT-L/4 \
--train-data-path /home/tongji209/majiawei/Dit/dataset/train/source \
--gt-data-path /home/tongji209/majiawei/Dit/dataset/train/target \
--results-dir inpaint-train-results \
--global-batch-size 128 \
--epochs 1400 
# --num-classes 0


