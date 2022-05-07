## Prepare

### To start with

To use codes in this directory, you need all submodules correctly configured. Every thing should be fine if you clone this repo with

```bash
git clone --recursive https://github.com/cure-lab/ContraNet
```

Otherwise, please use the following command

```bash
git submodule update --init --recursive
```

All the softlinks in `whitebox_attacks/lib` should locate their origins automatically.

### Prepare pre-trained cGAN

Download pretrained cGAN and classifier as indicated in [top-level readme.md](../README.md#prerequisites) to `whitebox_attacks/pretrain`, the structure is as follows

```
pretrain/
├── cifar10_adding_noise_cGAN_112000
│   ├── dis.pth
│   ├── model=E-current-weights-step=112000.pth
│   ├── model=G-current-weights-step=112000.pth
│   └── model=V-current-weights-step=112000.pth
├── classifier
│   ├── Cifar10_ResNet50_214_95.27_m.pth
│   ├── densenet169.pt
│   ├── gtsrb_ResNet18_E87_97.85.pth
│   └── MNIST_Net.pth
├── gtsrb_cGAN
│   ├── dis.pth
│   ├── model=E-current-weights-step=90000.pth
│   ├── model=G-current-weights-step=90000.pth
│   └── model=V-current-weights-step=90000.pth
└── MNIST_cGAN
    ├──dis.pth
    ├── model=D-current-weights-step=14000.pth
    ├── model=E-current-weights-step=14000.pth
    ├── model=G-current-weights-step=14000.pth
    └── model=V-current-weights-step=14000.pth
```

## Training of DMM

If you want to train our DMM model, you can follow the procedure as follows. If you only want to test our methods, we have provided pre-trained weight, please check next section for usage.

We provide pre-trained model and log for better understanding how our code works: [Google Drive](https://drive.google.com/drive/folders/1ewSEZOx8kIkeavs62AWjnFmHo8un_L_q?usp=sharing)

The directory structure should be as follows:

```
whitebox_attacks/
├── results
|   ├── MobileNetV2-cifar10-2021-04-10-11-21-11
│   │       ├── MobileNetV2.pth
│   │       ├── MobileNetV2_bestE3acc95.82.pth
│   │       ├── args.json
│   │       ├── lpmlpMix-10.log
│   │       └── tensorboard
│ 	└── cifar10AEPgd2
│           └── MobileNetV2-dense.01E.2_112000cifar10CondPgd-2021-05-24-12-35-50
│               ├── MobileNetV2.pth
│               ├── MobileNetV2_AEbestE91V97.48.pth
│               ├── MobileNetV2_bestE77V92.18.pth
│               ├── [base]MobileNetV2-dense.01E.2_112000cifar10CondPgd-2021-05-18-20-47-08
│               ├── args.json
│               ├── lpmlpMix-Dense.01PgdE.2_112000.log
│               └── tensorboard
└── pretrain
    └── done
        ├── MobileNetV2-256_32_2cifar10Gen-2021-04-08-21-32-16
        └── MobileNetV2-512_32_3cifar10-2021-04-08-21-03-43
```

The whole training procedures of DMM are as follows:

1. train feature extractor with deep metric loss for original image
    - run `lpDMLpretrain.py`
    - results and logs are in `pretrain/done/MobileNetV2-512_32_3cifar10-2021-04-08-21-03-43`
    - please check `pretrain/done/MobileNetV2-.../args.json` for command args

2. train feature extractor with deep metric loss for generated image
    - run `lpDMLpretrain.py`
    - results and logs are in `pretrain/done/MobileNetV2-256_32_2cifar10Gen-2021-04-08-21-32-16`
    - please check `pretrain/done/MobileNetV2-.../args.json` for command args

3. load the weights from previous steps and train a MLP
    - run `lpmlpMix.py`
    - results and logs are in `results/MobileNetV2-cifar10-2021-04-10-11-21-11`
    - please check `results/MobileNetV2-.../args.json` for command args

4. to further improve the performance, we also use AE for fine-tuning.
    - run `lpmlpMix.py` with `--useAE AE`
    - results and logs are in `results/cifar10AEPgd2/MobileNetV2-dense.01E.2_112000cifar10CondPgd-2021-05-24-12-35-50`
    - please check `results/cifar10AEPgd2/MobileNetV2-.../args.json` for command args
    - Note: due to time limit of our GPU server, we have to resume after 50 epochs.

The weight file from step4 is the final model weight we used for test.
Md5sum for `MobileNetV2_AEbestE91V97.48.pth`: `232802a68f57da260c0e6b8d7dc87c5b`.

## Whitebox Test

The procedures to test the performance with whitebox AEs are as follows

1. Download pretrained DMM to `./results`

```
results/
├── cifar10AEPgd2
│   └── MobileNetV2-dense.01E.2_112000cifar10CondPgd-2021-05-24-12-35-50
│       └── MobileNetV2_91V97.48.pth
├── gtsrbAEPgd
│   └── MobileNetV2-.01gtsrbCondPgd-2021-05-04-15-12-58
│       └── MobileNetV2_33V98.08.pth
└── MnistAEPgd
    └── MobileNetV2-C.01MNISTCondPgd-2021-05-06-15-04-16
        └── MobileNetV2_59V99.92.pth
```

2. Setup Python environment as `environment.yaml`.
3. Generate whitebox samples. See `scripts/gen_whitebox_sample.sh`. <br>**Note that** the output information is **NOT** the final results because we didn't set correct threshold here. If you have other generated AEs, you may skip this step, but please make sure to keep the data format the same as ours.

4. Plot ROC curve and get the threshold at `FPR@95%`. see `scripts/roc_threshold.sh`.

5. Finally, TPR and Detector's Accuracy can be tested through `scripts/test_whitebox.sh`. Furthre, AutoAttack whitebox results can be tested with `test/eval_detectorDictAA.py`.
