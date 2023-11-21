CUDA_VISIBLE_DEVICES=0,1 nohup python3 train.py \
--model='segvit_tiny' --batch-size=256 --data-path='~/../../data/sb56/data/Imagenet/' --output_dir='outputs/' \
 > logs/seg_vit_test.log 2>&1 &