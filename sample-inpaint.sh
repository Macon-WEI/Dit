python3 sample-inpaint.py --model DiT-S/4 \
--image-size 256 \
--result-dir ./sample-inpaint-results \
--seed 0 \
--num-sampling-steps 2 \
--sample-visual-every 1 \
--ckpt /home/tongji209/majiawei/Dit/Dit/inpaint-train-results/038-DiT-S-4/checkpoints/final.pt \
--predict-xstart \
--diffusion-steps 2