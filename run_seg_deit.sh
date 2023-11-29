CUDA_VISIBLE_DEVICES=0 nohup python3 train.py \
--model='segvit_tiny' --output_dir='outputs/' --num_tokens=196 --segmentation="slic" --batch-size=256 --data-set='CIFAR' --data-path='/home/sk138/data/' \
 > logs/timing_test_singleGPU.log 2>&1 &