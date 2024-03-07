# Construction Stage Classification
Construction stages classification

## Dataset 
The dataset consists of 4 cameras. Each camera captures 10 sequences (structures) such as each sequence has 87 timestamps
However, not all of these images are labelled. The ones without labels will be removed from dataset

## Splitting dataset
Generally, we can split them in ratio of 6-2-2 or 8-1-1. In this assigment, I want to use the latter approach since I believe that we'd have too few images for training if applying 6-2-2.
For each camera I randomly choose 8 sequence for training, 1 for testing and 1 for validation. Overall, I use 36 sequences for training, 4 for validation and 4 for testing 

## Task 1
I used RestNet50 as the backbone and only add a linear layer at the end to adjust to number of classes in our case
### Run the training
```python train.py --train --num_epoch 200 --batch_size 4 --lr 0,005```
### Run the evaluation to see how well the model performs on test set. 
The accuracy = number of accurate predictions / number of images in test set. I trained for 300 epochs and get the accuray of 99.62%

```python train.py --test```

### Test a specific image from the test set 
```python test.py --idx 1```

## Task 2
Unfortunately I am not able to finish the task. My idea is to utilize the Vision transformer architecture to solve the task. Specifically, we take e.g 25 images from the sequence then consider that them as patches of a bigger image. It is similar to ViT dividing image into small patches, just different that here we are given the patches already. For the positional encoding in ViT we can also do the same for our task here. Thanks to the given timestamps we can have the position of each image in the sequence ans therefore obtain the embedded position. Nevertheless, I haven't been able to put the idea into work since the adjustment for the shape in each layer is more complex than I expected
## References
https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
