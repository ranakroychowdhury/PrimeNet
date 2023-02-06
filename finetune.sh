python3 finetune.py --niters 2 --lr 0.0001 --batch-size 128 --rec-hidden 128 --n 8000 --quantization 0.016 \
--save 1 --classif --num-heads 1 --learn-emb --dataset physionet --seed 0 --task classification \
--pretrain_model 87623 --pooling ave --dev 0