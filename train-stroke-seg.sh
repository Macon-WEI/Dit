# export CUDA_LAUNCH_BLOCKING=1
python -m torch.distributed.run --nnodes=1 \
--nproc_per_node=1 \
./train_options/train_stroke_seg.py \
--model DiT-S/2 \
--character-path /home/tongji209/majiawei/stroke_segmentation/train_debug/characters-1000 \
--stroke-path /home/tongji209/majiawei/stroke_segmentation/train_debug/strokes-1000 \
--csv-path /home/tongji209/majiawei/stroke_segmentation/train_debug/stroke_data_1000.csv \
--results-dir stroke-seg-train-results \
--log-every 100 \
--visual-every 100 \
--global-batch-size 64 \
--num-workers 1 \
--learning-rate 1e-4 \
--epochs 1000 \
--predict-xstart \
--diffusion-steps 500
