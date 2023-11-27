CUDA_VISIBLE_DEVICES=0,1 nohup python3 train.py \
--model='segvit_tiny' --data-path='~/../../data/sb56/data/Imagenet/' --output_dir='outputs/' --num_tokens=-1 --segmentation="felz" --batch-size=16 \
 > logs/seg_vit_test2.log 2>&1 &