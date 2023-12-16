CUDA_VISIBLE_DEVICES=0 setsid nohup python train_vit.py --model segvit_tiny --batch-size 256 --n_points 768 \
    --zero-augments True --no-repeated-aug --mixup 0 --cutmix 0 --data-set CIFAR-SEG --data-path /home/sk138/data/cifar-100-python-segmented/cifar-196-BoW-seg/ \
    --output_dir /home/sk138/seg_deit/saves/BoW-seg-tiny --bag_of_words True --num_workers 12 \
    > /home/sk138/seg_deit/logs/segvit_tiny_BoW_seg2.txt 2>&1