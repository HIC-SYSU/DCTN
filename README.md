# # DCTN:Deformable Constraint Transport Network

## Introduction
This is a release of the paper： A Deformable Constraint Transport Network for Optimal Aortic Segmentation from CT Images, including the pre-processing, the network structure and the training implementation.

## News

* [10/23 2023] The code is open-sourced
* [10/30 2023] These files are updated
* [10/31 2023] The code can be available at https://github.com/HIC-SYSU and https://github.com/HICL-SYSU/DCTN.git.

## Requirement

*  torch == 1.7  

*  python == 3.8

*  SimpleITK == 2.2
  
*  (Optional) The code allows to load images in Nifti formats. If you wish to use nifti formats you should install SimpleITK
  
*  ....


##  Code List
* Pre-processing

* Dataloader 

* Model

* Training

## Results

These images depicts some results included in the DCTN papers.

* Some visualization results of the proposed network under different aortic cases.
  
![image](https://github.com/HIC-SYSU/DCTN/tree/main/IMG/3D.png)

* Some quantitative measurement analysis of the proposed network under different aortic cases.

![image](https://github.com/HIC-SYSU/DCTN/tree/main/IMG/B-A.png)
