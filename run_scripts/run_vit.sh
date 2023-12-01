CUDA_VISIBLE_DEVICES=0,1,2 nohup python -m torch.distributed.launch --nproc_per_node=3 --use_env train_vit.py --model deit_tiny_patch16_224 --batch-size 256 \
    --zero-augments True --no-repeated-aug --mixup 0 --cutmix 0 --data-set CIFAR --data-path /home/sk138/data/ --output_dir /home/sk138/seg_deit/saves/ \
    > /home/sk138/seg_deit/logs/vit_test.txt 2>&1