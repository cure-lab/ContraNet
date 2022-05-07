# ContraNet
The training code of ContraNet is in `./cifar10_ContraNet`.

The testing code of ContraNet against _white-box_ attacks are in `./whitebox_attacks` and _adaptive attacks_ in `./adaptive_attacks`.

**Prerequisties:**
Install necessay dependencies listed in ` environment.yaml`


**Pretained models:**

_Similarity Measure Model:_

dis: https://drive.google.com/file/d/1XOT_kyrJTwbs78vdWMJFNLl2lGoZa9az/view?usp=sharing

DMM: https://drive.google.com/file/d/19qJdRq05X4vR60y3SLk32X-NYUQkfMM7/view?usp=sharing

_cGAN:_

Encoder:https://drive.google.com/file/d/1U5F2UsKSX67mJ-hU4rh1-AZgUXCPDf0G/view?usp=sharing 
https://drive.google.com/file/d/1PmGwrB1eODsiQQu8TPad4oskolIoveMY/view?usp=sharing

generator:https://drive.google.com/file/d/1PueCACxOCh6-wdiss3BHBL021VjPdCwv/view?usp=sharing

_classifiers:_

Densenet169:https://drive.google.com/file/d/1kK-2wlu5xgS-iV6R5cGBG_Zyc7wwD4O9/view?usp=sharing


# Whitebox Test
0. `cd whitebox_attacks`.
1.  Download pretrained cGAN and classifier to `./pretrain`
    ```
    pretrain/
    ├── cifar10_adding_noise_cGAN_112000
    │   ├── dis.pth
    │   ├── model=E-current-weights-step=112000.pth
    │   ├── model=G-current-weights-step=112000.pth
    │   └── model=V-current-weights-step=112000.pth
    ├── classifier
    │   ├── densenet169.pt

    ```
2. Download pretrained DMM to `./results`
    ```
    results/
    ├── cifar10AEPgd2
    │   └── MobileNetV2-dense.01E.2_112000cifar10CondPgd-2021-05-24-12-35-50
    │       └── MobileNetV2_91V97.48.pth

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




# ContraNet against adaptive attacks.
0. `cd adaptive_attacks`
1. Download pretrained models to `./pretrain`
2. Download classifier `densenet169.pt` to `./`
3. For PGD adaptive attacks, run:
  ```
  python adaptive_targeted_PGD_linf.py [--adaptive_PGD_loss all| ssim_dis_dml| dis_dml| ssim_dis| ssim_dml| dis| dml| ssim]
  ```
4. For ATC+ContraNet against PGD adaptive attacks, run:
```
  python ATC_ContraNet/robust_classifier_adaptive_targeted_PGD_linf.py 
```
5. For OrthogonalPGD attack, run:
```
  python OrthogonalPGD/contraNetattack.py [--fpr 5|50] [--attack_iteration 200|40] [--adaptive_PGD_loss all| ssim_dis_dml| dis_dml| ssim_dis| ssim_dml| dis| dml| ssim]
```
6. ContraNet+ATC against OrthogonalPGD, run:
```
  python OrthogonalPGD/robust_classifier_adaptive_targeted_PGD_linf.py  [--fpr 5|50] [--attack_iteration 200|40] [--adaptive_PGD_loss all| ssim_dis_dml| dis_dml| ssim_dis| ssim_dml| dis| dml| ssim]
```
7. For C&W adaptive attacks, run:
```
  python targeted_cw_adaptive_attack.py
```




# Training 
0. ` cd cifar10_ContraNet`
1. step 1: Train the cGAN component of ContraNet:
 ```
 python adding_noise_main.py 
 ```
 Note that, you may turn off the --resume option if you want to train the model from scratch. After step 1, the basic version of ContraNet's generator part is done. Step 2 aimming to further improve the quality of the synthesis, one may skip this step.
 Our cGAN's implementation is based on https://github.com/POSTECH-CVLab/PyTorch-StudioGAN, one may refer to this repo for more instructions.
2. step 2 (optional): Train the second discriminator to help the cGAN generating synthesis more faithful to the input image.
 ```
 python mydiscriminator_main.py
 ```
 Once the second discriminator is done, finetune the cGAN model with the obtained second discriminator as an additional objective item by changing the ` adding_noise_worker.py` to `worker_train_d2D.py`. Then run:
 ```
 python adding_noise_main.py
 ```
 3. step 3: Train the Dis component in the similarity measurement model.
 ```
 python noisecGAN_adding_bengin_noise_augmentation_using_discrimator_as_dml.py
 ```
 4. step 4: Train the DMM component in the similarity measurement model.
 ```
 cd whitebox_attacks
 ```
 First, train the feature extractor part of DMM:
 ```
 python lpDMLpretrain.py
 ```
 Then, train the MLP part of DMM with the fixed feature extractor model:
 ```
 python lpmplMix.py
 ```
