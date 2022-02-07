#!/bin/bash

#>>>>>>>>>>>>>>>>>> AA gen <<<<<<<<<<<<<<<<<<

# WideResNet-70-16
# python -u tools/gen_autoattack.py
# python -u tools/gen_autoattack.py --model Rebuffi2021Fixing_70_16_cutmix_ddpm

# WideResNet-28-10
# python -u tools/gen_autoattack.py --model Rebuffi2021Fixing_28_10_cutmix_ddpm
# python -u tools/gen_autoattack.py --model Sehwag2020Hydra

#>>>>>>>>>>>>>>>>>>> whitebox eval <<<<<<<<<<<<<<<<<<<<<<<<<<

# share_parama="-c config/configsCifar10GDnoise112000.json \
# --pretrain results/cifar10AEPgd2/MobileNetV2-dense.01E.2_112000cifar10CondPgd-2021-05-24-12-35-50/MobileNetV2_91V97.48.pth \
# --thresh  0 0 --log_dir results/log/whitebox/cifar10AEPgd295/E.2AEV97.48Dnoise112000"

# python tools/gen_whitebox.py --attack BIM ${share_param} && \
# python tools/gen_whitebox.py --attack BIML2 ${share_param} && \
# python tools/gen_whitebox.py --attack PGD ${share_param} && \
# python tools/gen_whitebox.py --attack PGDL2 ${share_param} && \
# python tools/gen_whiteboxDF.py --attack DF ${share_param} && \
# python tools/gen_whiteboxDF.py --attack DFL2 ${share_param} && \
# python tools/gen_whiteboxCW.py --attack CW ${share_param} && \
# python tools/gen_whiteboxCW.py --attack CWinf ${share_param} && \
# python tools/gen_whiteboxEADA.py --attack EADA ${share_param} && \
# python tools/gen_whiteboxEADA.py --attack EADAL1 ${share_param}

