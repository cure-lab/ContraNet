#!/bin/bash

#CIFAR10
share_param="--log_dir results/log/roc/dis_ssim_100p \
--pretrain results/cifar10AEPgd2/MobileNetV2-dense.01E.2_112000cifar10CondPgd-2021-05-24-12-35-50/MobileNetV2_91V97.48.pth \
--csv_file results/log/roc/roc_100p.csv \
--load_dir results/log/roc/dis_ssim_100p"

python -u tools/roc_curve.py --arch ${arch} ${share_param}


# MNIST
share_param="--log_dir results/log/roc/mnist_ssim_100p \
--pretrain results/MnistAEPgd/MobileNetV2-C.01MNISTCondPgd-2021-05-06-15-04-16/MobileNetV2_59V99.92.pth \
--csv_file results/log/roc/mnist_roc_100p.csv \
--load_dir results/log/roc/mnist_ssim_100p"

python -u tools/roc_curve.py --arch ContraNet2_3ssH \
-c config/configsMnist.json ${share_param}



# GTSRB
share_param="--log_dir results/log/roc/gtsrb_ssim_100p \
--pretrain results/gtsrbAEPgd/MobileNetV2-.01gtsrbCondPgd-2021-05-04-15-12-58/MobileNetV2_33V98.08.pth \
--csv_file results/log/roc/gtsrb_roc_100p.csv \
--load_dir results/log/roc/gtsrb_ssim_100p"

python -u tools/roc_curve.py --arch ContraNet2_3ssH \
-c config/configsGTSRB.json ${share_param}
