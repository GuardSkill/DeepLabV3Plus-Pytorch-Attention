python3 main.py --divide_data 1000 --data_root /home/lulu/Dataset/ --model deeplabv3plus_mobilenetSA --gpu_id 2,3 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16
# train network with 1000 test samples from training set      9582 training 1449 eva 1000test
# python main.py --divide_data 1000 --data_root /home/lulu/Dataset/ --model deeplabv3plus_mobilenet  --gpu_id 2 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16
#
# train network with original dataset         10582 training 1449 eva
# python main.py --divide_data 0 --data_root /home/lulu/Dataset/ --model deeplabv3plus_mobilenet --vis_port 28333 --gpu_id 2 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16
#
 # evaluate network/test  os=16
# python main.py --divide_data 1000 --data_root /home/lulu/Dataset/ --test_only ---model deeplabv3plus_mobilenet  --gpu_id 0 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --ckpt checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth  --save_val_results
# python main.py --divide_data 1000 --data_root /home/lulu/Dataset/ --test_only --test_divide --model deeplabv3plus_mobilenet  --gpu_id 2 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --ckpt checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth --save_val_results
#
# SA training:       --model deeplabv3plus_mobilenetSA
# python main.py --divide_data 1000 --data_root /home/lulu/Dataset/ --model deeplabv3plus_mobilenetSA --gpu_id 2 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16
# divide dataset   --divide_data 1000
# visualization: --enable_vis --vis_port 28333
# dataset root: --data_root /home/lulu/Dataset/