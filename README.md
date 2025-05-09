# SlowFashionProject
Cloth Classification for the company slow fashion. All data and model trained on it belong to slow fashion and cannot be published.


# Dataset to explore

For quick testing  
[kaggle Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)  
Bigger dataset (7gigs)  
[kaggle clothing dataset](https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full)  
Biggest dataset (250gigs)  
[DeepFashion2](https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok), [Form](https://docs.google.com/forms/d/e/1FAIpQLSeIoGaFfCQILrtIZPykkr8q_h9qQ5BoTYbjvf95aXbid0v2Bw/viewform) to get the code

⚠️ DeepFashion is not available for commercial use, [source](https://docs.google.com/forms/d/e/1FAIpQLSeIoGaFfCQILrtIZPykkr8q_h9qQ5BoTYbjvf95aXbid0v2Bw/viewform), using it for SlowFashion production is copyright infrigement and future audit WILL see it.
![alt text](assets/image-1.png)
![alt text](assets/image-2.png)

# About the papers
Deep Learning for Clothing Style Recognition Using YOLOv5
This paper do Clothing Style Recognition with YOLOv5 and R-CNN, the metric used are average precison, mean average precison, recall, F1-score, model
size, and frame per second. They Evaluate different architecture and want to be deployable on mobile device.

DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations

Deep Residual Learning for Image Recognition
Best paper about residual neural network, should be better than VGG-16 but a lot deeper

DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations
presents the DeepFashion dataset, a large-scale dataset for benchmarking tasks in fashion recognition and retrieval. 
Measure:
1. Category and Attribute Prediction:
Top-k accuracy (e.g., Top-1, Top-5): Measures whether the ground truth label is among the top-k predicted labels.
Mean Average Precision (mAP): Especially for attribute prediction, to account for multi-label outputs.

1. Landmark Detection:
Normalized Error (NE): Distance between predicted and true landmark locations, normalized by image size.
Detection Rate @ ε: Percentage of correctly predicted landmarks under a specific error threshold ε.

1. Clothes Retrieval:
Top-k retrieval accuracy: Measures whether the correct match is among the top-k retrieved items.
Recall@k: Proportion of queries for which the correct item appears in the top-k results.