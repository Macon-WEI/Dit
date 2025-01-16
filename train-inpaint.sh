torchrun --nnodes=1 \
--nproc_per_node=4 \
./train_options/train_inpaint.py \
--model DiT-L/4 \
--train-data-path /remote-home/zhangxinyue/DiT/train/source \
--gt-data-path /remote-home/zhangxinyue/DiT/train/target \
--results-dir inpaint-train-results \
--global-batch-size 256 \
--epochs 1400 
# --num-classes 0


