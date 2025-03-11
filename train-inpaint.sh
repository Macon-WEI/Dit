torchrun --nnodes=1 \
--nproc_per_node=4 \
./train_options/train_inpaint.py \
--model DiT-S/4 \
--train-data-path /remote-home/zhangxinyue/DiT/train/source \
--gt-data-path /remote-home/zhangxinyue/DiT/train/target \
--results-dir inpaint-train-results \
--log-every 100 \
--global-batch-size 256 \
--learning-rate 1e-4 \
--epochs 1500 \
--predict-xstart \
--diffusion-steps 100
