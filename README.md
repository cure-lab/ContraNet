
# Whitebox Test
1. Download pretrained cGAN and classifier to `./pretrain`
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
2. Download pretrained DMM to `./results`
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
3. Setup Python environment as `environment.yaml`.
4. Generate whitebox samples. See `scripts/gen_whitebox_sample.sh`. **Note
   that** the output information is not the final results because we didn't set
   correct threshold here.

5. Plot ROC curve and get the threshold at `FPR@95%`. see
   `scripts/roc_threshold.sh`.

6. Finally, TPR and Detector's Accuracy can be tested through
   `scripts/test_whitebox.sh`. Furthre, AutoAttack whitebox results can be
   tested with `test/eval_detectorDictAA.py`.
