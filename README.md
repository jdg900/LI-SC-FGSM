# Strengthening the Transferability of Adversarial Examples Using Advanced Looking Ahead and Self-Cutmix (CVPRW 2022)

This repository is the official PyTorch implementation of paper:

>[Strengthening the Transferability of Adversarial Examples Using Advanced Looking Ahead and Self-CutMix ](https://openaccess.thecvf.com/content/CVPR2022W/ArtOfRobust/papers/Jang_Strengthening_the_Transferability_of_Adversarial_Examples_Using_Advanced_Looking_Ahead_CVPRW_2022_paper.pdf)
>
>Donggon Jang* (KAIST), Sanghyeok Son∗ (KAIST), and Dae-Shik Kim (KAIST) (*The authors have equally contributed.)
>
>Accept to CVPRW 2022 (The Art of Robustness: Devil and Angel in Adversarial Machine Learning)

>**Abstract:** Deep neural networks (DNNs) are vulnerable to adversarial examples generated by adding malicious noise imperceptible to a human. The adversarial examples successfully fool the models under the white-box setting, but the performance of attacks under the black-box setting degrades significantly, which is known as the low transferability
problem. Various methods have been proposed to improve transferability, yet they are not effective against adversarial training and defense models. In this paper, we introduce two new methods termed Lookahead Iterative Fast Gradient Sign Method (LI-FGSM) and Self-CutMix (SCM) to address the above issues. LI-FGSM updates adversarial perturbations with the accumulated gradient obtained by looking ahead. A previous gradient-based attack is used for looking ahead during N steps to explore the optimal direction at each iteration. It allows the optimization process to escape the sub-optimal region and stabilize the update directions. SCM leverages the modified CutMix, which copies a patch from the original image and pastes it back at random positions of the same image, to preserve the internal information. SCM makes it possible to generate
more transferable adversarial examples while alleviating the overfitting to the surrogate model employed. Our two
methods are easily incorporated with the previous iterative gradient-based attacks. Extensive experiments on ImageNet
show that our approach acquires state-of-the-art attack success rates not only against normally trained models but also against adversarial training and defense models.

## Requirements

To install requirements:
```setup
conda env create -n [your env name] -f environment.yaml
conda activate [your env name]
```

## Pre-trained Models
To run the code, you should download pretrained models and the data. Please place pretrained models and the data under the models/ directory and dev_data/, respectively.  

You can download pretrained models here:

- [Pre trained models](https://drive.google.com/drive/folders/1oE2ead9ryKY6MB-0JO8U5lnGAoee-WXW?usp=sharing) trained on Inc-v3, Inc-v4, IncRes-v2, Res-101. 

## Dataset

You can download the dataset here:

- [Dataset](https://drive.google.com/drive/folders/1xfUOPuynX-2GaHt-miHT9z0RepG0gKA8?usp=sharing) is sampled from ImageNet val set. 


## Attack

To generate adversarial examples according to the paper, run this command:

To generate adversarial exmples of SCM-P using MI-FGSM
```
CUDA_VISIBLE_DEVICES=[gpu id] python mi_scp_fgsm.py --batch_size 10 --output_dir ./mi_scp_fgsm
```
To generate adversarial exmples of SCM-R using MI-FGSM
```
CUDA_VISIBLE_DEVICES=[gpu id] python mi_scr_fgsm.py --batch_size 10 --output_dir ./mi_scr_fgsm
```
To generate adversarial exmples of LI-FGSM
```
CUDA_VISIBLE_DEVICES=[gpu id] python li_fgsm.py --batch_size 10 --output_dir ./li_fgsm
```
To generate adversarial exmples of LI-SCP-CT-FGSM (If you encounter CUDA out of memory issue, reduce the batch_size)
```
CUDA_VISIBLE_DEVICES=[gpu id] python li_scp_ct_fgsm.py --batch_size 2 --output_dir ./li_scp_ct_fgsm
```
To generate adversarial exmples of LI-SCR-CT-FGSM (If you encounter CUDA out of memory issue, reduce the batch_size)
```
CUDA_VISIBLE_DEVICES=[gpu id] python li_scr_ct_fgsm.py --batch_size 2 --output_dir ./li_scr_ct_fgsm
```

## Evaluation

To evaluate the attack sucess rates on 4 normally trained models (Inc-v3, Inc-v4, IncRes-v2, Res-101) and 3 adversarially trained models (Inc-v3(ens3), Inc-v3(ens4), IncRes-v2(ens)), run this command:

```eval
CUDA_VISIBLE_DEVICES=[gpu id] python simple_eval.py --input_dir [path of saved adversarial images]
```

## Results

Our method achieves the following attack sucess rates on :

| Model  | Attack  | Inc-v3 | Inv-v4 | IncRes-v2 | Res-101 | Inc-v3 (ens3) | Inc-v3 (ens4) | IncRes-v2 (ens) |  
| -------| --------|--------|--------| ----------| -------| ---------------| --------------| --------------- |  
| Inc-v3 | MI-SCP-FGSM | 100.00% | 87.28% | 85.04% | 80.92% | 50.16% | 45.28% | 27.36% |
| Inc-v3 | MI-SCR-FGSM | 99.85% | 85.90% | 83.68% | 81.50% | 62.58% | 58.94% | 36.86% | 
| Inc-v3 | LI-FGSM | 100.00% | 91.02% | 88.28% | 80.66% | 40.14% | 38.84% | 22.00% |   
| Inc-v3 | LI-SCP-CT-FGSM | 100.00% | 98.70% | 97.87% | 95.80% | 91.43% | 88.90% | 78.83% |  
| Inc-v3 | LI-SCR-CT-FGSM | 99.90%  | 97.13% | 96.50% | 92.83% | 89.90% | 88.27% | 77.00% | 


## Acknowledgements
This code is built on [SI-NI-FGSM](https://github.com/JHL-HUST/SI-NI-FGSM). We thank the authors for sharing their codes of SI-NI-FGSM.

## Contact
If you have any questions, feel free to contact me (jdg900@kaist.ac.kr)
