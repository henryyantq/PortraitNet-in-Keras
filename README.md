# PortraitNet-in-Keras
This is a Keras implementation of PortraitNet (C&amp;G 2019). Dataset can be download here: [EG1800](https://goo.gs/pqs8e)

The visualized accuracy & loss history of my training process (Laptop: AMD Ryzen 7 5800H + RTX 3060 115W + 16GB DDR4 2933MHz):

Accuracy:

![accu](https://user-images.githubusercontent.com/20149275/204752651-a241d480-71f5-467c-8fb8-2c171158b844.png)

Loss:

![loss](https://user-images.githubusercontent.com/20149275/204752676-0121c760-c2af-4773-b9e3-7d5360139dd5.png)

The model structure is taken from the original paper:

![image](https://user-images.githubusercontent.com/20149275/204755042-72a3d0a4-21a6-4174-a094-7ab85e5e01c7.png)

*Note that my implementation is for 128x128 RGB images as inputs, rather than 224x224 as is mentioned in the paper, which is only for faster training and inference procedures. If you're familiar with U-Net, MobileNetV2, residual block, etc., the PortraitNet should be easy for you to understand. Moreover, if you have some experience using Keras, I believe my code is one of the most intuitive versions for reference (pn2.py or PortraitNet.ipynb). 
