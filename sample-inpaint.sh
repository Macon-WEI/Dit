python3 sample-inpaint.py --model DiT-S/4 \
--image-size 256 \
--result-dir ./sample-inpaint-results \
--seed 0 \
--num-sampling-steps 10 \
--sample-visual-every 1 \
--ckpt /remote-home/zhangxinyue/DiT/Dit/inpaint-train-results/018-DiT-S-4/checkpoints/best.pt \
--predict-xstart \
--diffusion-steps 20