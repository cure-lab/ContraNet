# ContraNet against adaptive attacks.
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
