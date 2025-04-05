python3 real-sample.py --model DiT-S/2 \
--image-size 256 \
--result-dir /home/tongji209/majiawei/Dit/dataset/real-sample \
--test-path /home/tongji209/majiawei/Dit/dataset/test/task1/eroded \
--seed 0 \
--num-sampling-steps 10 \
--sample-visual-every 1 \
--ckpt /home/tongji209/majiawei/Dit/Dit/inpaint-train-results/052-DiT-S-2/checkpoints/best.pt \
--predict-xstart \
--diffusion-steps 100