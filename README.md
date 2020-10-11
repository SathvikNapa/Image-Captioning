# Image-Captioning using Pytorch
 ------------------------------
 
 This repository contains a Deep Neural Network that automatically generates captions from Images.
 
# Network Architecure
 --------------------
 ![image](https://github.com/sathviksunny/Image-Captioning/blob/main/images/encoder-decoder.png)
 1. Encoder, A CNN-Encoder network that generates a feautre embedding vector from an image input
 ![image](https://github.com/sathviksunny/Image-Captioning/blob/main/images/encoder.png)
 
 2. Decoder, An LSTM integrated Sequential Neural Network which converts the embedding feature vector as a sequence of tokens in the form of sentences.
 ![image](https://github.com/sathviksunny/Image-Captioning/blob/main/images/decoder.png)
 
 # Instructions
 
 Coco-Dataset
 ------------
 
 ![image](https://github.com/sathviksunny/Image-Captioning/blob/main/images/coco-examples.jpg)
 
 1. Clone this repo: https://github.com/cocodataset/cocoapi  
```
git clone https://github.com/cocodataset/cocoapi.git  
```

2. Setup the coco API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI  
make  
cd ..
```

3. Download some specific data from here: http://cocodataset.org/#download (described below)
* Under **Annotations**, download:
  * **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
  * **2014 Testing Image info [1MB]** (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

* Under **Images**, download:
  * **2014 Train images [83K/13GB]** (extract the train2014 folder and place at location cocoapi/images/train2014/)
  * **2014 Val images [41K/6GB]** (extract the val2014 folder and place at location cocoapi/images/val2014/)
  * **2014 Test images [41K/6GB]** (extract the test2014 folder and place at location cocoapi/images/test2014/)

4. The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order (`0_Dataset.ipynb, 1_Preliminaries.ipynb, 2_Training.ipynb, 3_Inference.ipynb`).

5. The 'model.py' file contains the Encoder-Decoder Neural Network Architecture.

# Results
----------
![image](https://github.com/sathviksunny/Image-Captioning/blob/main/images/Result-1.png)
![image](https://github.com/sathviksunny/Image-Captioning/blob/main/images/Result-2.png)

