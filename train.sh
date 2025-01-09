torchrun --nnodes=1 \
--nproc_per_node=1 \
./train_options/train_original.py \
--model DiT-L/4 \
--data-path /home/tongji209/majiawei/Dit/train-data \
--results-dir train-results \
--global-batch-size 128 \
--resume-from-checkpoint /home/tongji209/majiawei/Dit/Dit/train-results/009-DiT-L-4/checkpoints/0150000.pt \
--epochs 600 \
--num-classes 1

# accelerate launch --mixed_precision fp16 ./train_options/train_original.py --model DiT-L/4 --data-path /home/tongji209/majiawei/Dit/train-data \
# --results-dir train-results \
# --num-classes 1