# Nuclear_Dataset_Segmentation


##  Introduction of Nuclear Dataset


## Getting the code
You can download all the files in this repository by cloning this repository:
```
https://github.com/Jia-Wei-Liao/Nuclear_Dataset_Segmentation.git
```


## Repository Structure
      .
      ├──checkpoint
      ├──dataset
      ├──src
      |   ├──trainer.py
      |   └──utils.py
      ├──config.py
      ├──inference.py 
      ├──make_annot.py
      └──train.py

## Requirements


## Download
- You can download the dataset on the Google Drive:  
<https://drive.google.com/drive/folders/1wYiUhk8yma6RJJ2RcjhIoLyCIy-J4lkU?usp=sharing>
- You can download the weight on the Google Drive:  
<https://drive.google.com/drive/folders/1BPxTCnvXPHck3hg5QOFD1xJlMDZplKfh?usp=sharing>  


## Dataset
```
python make_annot.py
```


## Training
To train the model, you can run this command:
```
python train.py --model mask_rcnn_R_50_C4_1x --device cuda:0
```
- model: mask_rcnn\_R\_{50, 101}\_{C4, FPN}\_{1, 3}x
- device: cpu or cuda:{0, 1, 2, 3}


## Inference
To inference the results, you can run this command:
```
python inference.py --checkpoint '2021-12-10-09-43' --weight_name model_final.pth
```


## Reproducing submission
To reproduce our submission, please do the following steps:
1. [Getting the code](https://github.com/Jia-Wei-Liao/Nuclear_Dataset_Segmentation#Getting-the-code)
2. [Install the package](https://github.com/Jia-Wei-Liao/Nuclear_Dataset_Segmentation#requirements)
3. Download the dataset and weight
4. [Inference](https://github.com/Jia-Wei-Liao/Nuclear_Dataset_Segmentation#inference)


## Results
| method    | mAP     | 
| --------- | ------- |
|           | 0.24374 |


## Reference
[1] https://github.com/facebookresearch/detectron2  
[2] https://detectron2.readthedocs.io/en/latest/tutorials/install.html 
