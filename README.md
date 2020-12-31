# Neural-Style-Transfer
Neural Style Transfer using VGG-19 Network

## Description:
Implementation of system proposed by Leon et al. in A Neural Algorithm of Artistic Style, 2015.

## Installation:
* Clone the repository

## Usage:
* ./main.py [--content] [--style] [--total_step=INT] [--log_step=INT] [--sample_step=INT] [--style_weight=FLOAT] [--lr=FLOAT]

## Results:
##### Initial attemps, with very wrong naming of the model features: 
![][/outputs/initial.jpg]

##### Fixed the naming according to the paper: 
* using conv features for both content and style with mse loss for both
![][/outputs/"mse and conv".jpg]

* using conv features for both content and style with l1 loss for both
![][/outputs/"l1 and conv".jpg]

* using conv5 for content and all relu features for the style with mse loss for both
![][/outputs/"conv5 and all relu with mse loss".jpg]

* using conv5 for content and all relu features for the style with l1 loss for both
![][/outputs/"conv5 with all relu with l1 loss".jpg]

#### Other outputs:
![][/outputs/"dancing starry night".jpg] ![][outputs/skyline.jpg]
![][/outputs/"starry night".jpg] ![][outputs/mulicolour.jpg]
![][/outputs/multicolourdancing.jpg]

## Try it yourself
### Resources:
* Paper: [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
* For understanding Gram Matrices: [Neural Style Transfer Tutorial -Part 1](https://towardsdatascience.com/neural-style-transfer-tutorial-part-1-f5cd3315fa7f)
* [Intuitive Guide to Neural Style Transfer](https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-neural-style-transfer-ef88e46697ee)
* [NEURAL TRANSFER USING PYTORCH](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
* https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/neural_style_transfer
* Rather helpful improvements to get more visually appealing results: 
  * [How to Get Beautiful Results with Neural Style Transfer](https://towardsdatascience.com/how-to-get-beautiful-results-with-neural-style-transfer-75d0c05d6489)
  * https://github.com/EugenHotaj/pytorch-generative
