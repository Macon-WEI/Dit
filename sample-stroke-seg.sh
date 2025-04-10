python3 sample-stroke-seg.py --model DiT-S/2 \
--image-size 256 \
--result-dir ./sample-stroke-seg-results \
--seed 0 \
--num-sampling-steps 10 \
--sample-visual-every 1 \
--ckpt /home/tongji209/majiawei/stroke_segmentation/Dit/stroke-seg-train-results/003-DiT-S-2/checkpoints/best.pt \
--diffusion-steps 500