torchrun --nnodes=1 \
--nproc_per_node=1 \
./train_options/train_inpaint.py \
--model DiT-S/2 \
--train-data-path /home/tongji209/majiawei/Dit/dataset/train/source \
--gt-data-path /home/tongji209/majiawei/Dit/dataset/train/target \
--results-dir inpaint-train-results \
--log-every 100 \
--visual-every 10 \
--global-batch-size 32 \
--num-workers 1 \
--learning-rate 1e-4 \
--epochs 1500 \
--predict-xstart \
--diffusion-steps 100
