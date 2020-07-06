# divide data command:
#python main.py --data_root /home/lulu/Dataset/cityscapes --model deeplabv3plus_mobilenet --dataset cityscapes --gpu_id 0,1  --lr 0.1 --crop_size 768 --batch_size 14 --output_stride 16
#  --gpu_id 0,1   --divide_data 375
#python main.py --data_root /home/lulu/Dataset/cityscapes --model deeplabv3plus_mobilenet --divide_data 375 --dataset cityscapes --gpu_id 0,1  --lr 0.1 --crop_size 768 --batch_size 14 --output_stride 16
#
#multiple model: --start_class 1 --total_itrs 5000
# python train_multiple_models.py --start_class 0 --total_itrs 5000 --divide_data 375 --data_root /home/lulu/Dataset/cityscapes --model deeplabv3plus_mobilenet_M --divide_data 375 --dataset cityscapes --gpu_id 2,3 --lr 0.01 --crop_size 768 --batch_size 14 --output_stride 16
#--ckpt checkpoints/latest_deeplabv3plus_mobilenet_cityscapes_os16.pth
# --test_only  --save_val_results
# --model deeplabv3plus_mobilenetSA
# --val_interval 1000
# visualization: --enable_vis --vis_port 28333
# dataset root: --data_root /home/lulu/Dataset/
#
#
# multiple
#  1-2: --batch_size 14 3-5:12   6-8:14  9-14:12  15-16:8  17::error  18-19:12
# --start_class 3 --batch_size 12
#python train_multiple_models.py --start_class 6 --divide_data 375 --total_itrs 5000 --data_root /home/lulu/Dataset/cityscapes --model deeplabv3plus_mobilenet_M --divide_data 375 --dataset cityscapes --gpu_id 2,3 --lr 0.01 --crop_size 768 --batch_size 14 --output_stride 16
#Test:
#  --save_val_results --alpha 0.5 --test_only --sin_model 'checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth'
#