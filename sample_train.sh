curl -LO http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz
rm -rf flower_photos/LICENSE*
echo "Dataset Downloaded and extracted! Ready for Training!"

python train_val_split.py --dataset flower_photos --val_split 0.2
echo "Training and Validation Set created"

python general_model.py --train train_dir --val val_dir --logs log_resnet --create_bottleneck --bottleneck_dir bottlenecks\
  --base_model resnet50 --bottlenecks_batch_size 128 --epochs 100 --weight_file Resnet50_top.h5\
  --batch_size_train 32 --lr 0.001 --bottleneck_tensorname activation_46
echo "Training Completed, now testing will start. It will just print given image, the class names and confidence."

python general_test.py --weight_file Resnet50_top.h5 --label_file essential_files/label_map.json \
 --img_dir val_dir/tulips --base_model resnet50
echo "If you get any problems feel free to open an issue in github repo!"
