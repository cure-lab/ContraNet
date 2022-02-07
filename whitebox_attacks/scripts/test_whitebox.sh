#!/bin/bash

share_param="--log_dir results/log/whitebox/ssim/cifar10AEPgd2 \
--pretrain results/cifar10AEPgd2/MobileNetV2-dense.01E.2_112000cifar10CondPgd-2021-05-24-12-35-50/MobileNetV2_91V97.48.pth \
--csv_file results/log/whitebox/ssim/cifar10.csv \
--drop_rate 0.0 --arch ContraNet2_3ssH "

python test/eval_detectorDict.py ${share_param} --thresh -1.334 0.0107 0.0551


# MNIST
share_param="--log_dir results/log/whitebox/ssim/mnistAEPgdC.0195/ \
--pretrain results/MnistAEPgd/MobileNetV2-C.01MNISTCondPgd-2021-05-06-15-04-16/MobileNetV2_59V99.92.pth \
--csv_file results/log/whitebox/ssim/mnist.csv \
-c config/configsMnist.json --drop_rate 0.0 \
--arch ContraNet2_3ssH "

python test/eval_detectorDict.py ${share_param} --thresh -2.2437 0.9994 0.3898


# GTSRB
share_param="--log_dir results/log/whitebox/ssim/gtsrbAEPgd95/ \
--pretrain results/gtsrbAEPgd/MobileNetV2-.01gtsrbCondPgd-2021-05-04-15-12-58/MobileNetV2_33V98.08.pth \
--csv_file results/log/whitebox/ssim/gtsrb.csv \
-c config/configsGTSRB.json --drop_rate 0.0 \
--arch ContraNet2_3ssH --batch_size 4 "

python test/eval_detectorDict.py ${share_param} --thresh -3.7287 0.9022 0.0753
