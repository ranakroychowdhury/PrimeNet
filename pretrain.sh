python3 pretrain.py --niters 2 --lr 0.0001 --batch-size 128 --rec-hidden 128 --n 8000 --quantization 0.016 \
--save 1 --classif --num-heads 1 --learn-emb --dataset physionet --seed 0 --add_pos --transformer \
--pooling bert --pretrain_tasks full2 --segment_num 3 --mask_ratio_per_seg 0.05 --dev 0
