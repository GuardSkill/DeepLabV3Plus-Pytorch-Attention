#
#   TEST for multiple mode: --alpha 0.5  --save_val_results --test_only  --sin_model 'checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth'
#
#
#  from class13: --total_itrs 5000 --loss_type logit --divide_data 1000 --data_root /home/lulu/Dataset/ --model deeplabv3plus_mobilenetSA_M --gpu_id 3 --year 2012_aug --crop_val --lr 0.001 --crop_size 513 --batch_size 18 --output_stride 16
#
#multiple models from class15-20  & 1-12  (batch==16):
#python train_multiple_models.py --total_itrs 5000 --loss_type logit --divide_data 1000 --data_root /home/lulu/Dataset/ --model deeplabv3plus_mobilenet_M --gpu_id 3 --year 2012_aug --crop_val --lr 0.001 --crop_size 513 --batch_size 16 --output_stride 16
#python train_multiple_models.py --total_itrs 5000 --loss_type logit --divide_data 1000 --data_root /home/lulu/Dataset/ --model deeplabv3plus_mobilenetSA_M --gpu_id 2 --year 2012_aug --crop_val --lr 0.001 --crop_size 513 --batch_size 16 --output_stride 16
