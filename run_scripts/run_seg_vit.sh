CUDA_VISIBLE_DEVICES=0,1 setsid nohup python -m torch.distributed.launch --nproc_per_node=2 --use_env train_vit.py --model segvit_tiny --batch-size 256 \
    --zero-augments True --no-repeated-aug --mixup 0 --cutmix 0 --data-set CIFAR-SEG --data-path /home/sk138/data/cifar-100-python-segmented/cifar-196-64/ \
    --output_dir /home/sk138/seg_deit/saves/ --num_workers 12 --resume /home/sk138/seg_deit/saves/checkpoint.pth --start_epoch 7 \
    > /home/sk138/seg_deit/logs/segvit_tiny_cifar100_rfft.txt 2>&1