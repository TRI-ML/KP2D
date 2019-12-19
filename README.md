# Neural Outlier Rejection for Self-Supervised Keypoint Learning
Code coming soon.

## Overview
![](media/imgs/diagram_architecture.png)
- **IO-Net:** A novel proxy task for the self-supervision of keypoint description. 
- **KeyPointNet:**  An improved keypoint-network architecture that is especially amenable to robust keypoint detection and description.

[**[Full paper]**](https://openreview.net/pdf?id=Skx82ySYPH)

## Over-fitting Examples

- **Toy example:**
<p align="center">
  <img src="media/gifs/v1.gif" alt="Target Frame" width="230" />
  <img src="media/gifs/h1.gif" alt="Heatmap" width="230" />
  <img src="media/gifs/w1.gif" alt="Source Frame" width="230" />
</p>

- **TRI example:**
<p align="center">
  <img src="media/gifs/compressed_v2.gif" alt="Target Frame" width="230" />
  <img src="media/gifs/compressed_h2.gif" alt="Heatmap" width="230" />
  <img src="media/gifs/compressed_w2.gif" alt="Source Frame" width="230" />
</p>

## Qualatitive Results

- **Illumination Cases:**

<p align="center">
  <img src="media/imgs/l1.png" alt="Illumination case(1)" width="600" />
  <img src="media/imgs/l2.png" alt="Illumination case(2)" width="600" />
</p>

- **Perspective Cases:**
<p align="center">
  <img src="media/imgs/p1.png" alt="Perspective case(1)" width="600" />
  <img src="media/imgs/p2.png" alt="Perspective case(2)" width="600" />
</p>

- **Rotation Cases:**
<p align="center">
  <img src="media/imgs/r1.png" alt="Rotation case(1)" width="600" />
  <img src="media/imgs/r2.png" alt="Rotation case(2)" width="600" />
</p>

## Citation
Please use the following citation when referencing our work:
```
@inproceedings{
tang2020neural,
title={Neural Outlier Rejection for Self-Supervised Keypoint Learning},
author={Jiexiong Tang and Hanme Kim and Vitor Guizilini and Sudeep Pillai and Rares Ambrus},
booktitle={International Conference on Learning Representations},
year={2020}
}
```
