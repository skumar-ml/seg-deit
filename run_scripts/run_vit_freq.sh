CUDA_VISIBLE_DEVICES=0 nohup python train_vit.py --model deit_tiny_patch16_224 --batch-size 256 --zero-augments True --no-repeated-aug --mixup 0 --cutmix 0\
    --data-set CIFAR --data-path /home/sk138/data/ --frequency True --output_dir /home/sk138/seg_deit/saves/vit_freq_phasemag \
    > /home/sk138/seg_deit/logs/vit_freq_phasemag.txt 2>&1