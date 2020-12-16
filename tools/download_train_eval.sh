#!/bin/bash
set -ex
wget -P models --no-clobber https://storage.googleapis.com/models-hao/mobilenet-v1-ssd-mp-0_675.pth
python open_images_downloader.py --class_names="Helmet" --root "data/open_images" --num_workers=50
python train_ssd.py --dataset_type open_images --datasets data/open_images \
  --net mb1-ssd --pretrained_ssd models/mobilenet-v1-ssd-mp-0_675.pth \
  --scheduler cosine --lr 0.01 --base_net_lr 0.01 --validation_epochs 1 \
  --num_epochs 1 --batch_size 32 --debug_steps 1
python eval_ssd.py --dataset_type open_images --net mb1-ssd \
  --trained_model models/mb1-ssd-Epoch-0*.pth \
  --label_file models/open-images-model-labels.txt --dataset data/open_images
