# Construction Stage Classification
Construction stages classification

## Dataset 
The dataset consists of 4 cameras. Each camera captures 10 sequences/structures such as each sequence has 87 timestamps
However, not all of these images are labelled. The ones without labels will be removed from data loader

## Splitting dataset
Generally, we can split them in ratio of 6-2-2 or 8-1-1. In this assigment, I want to use the latter approach since I believe that we'd have too few images for training if applying 6-2-2.
For each camera I randomly choose 8 sequence for training, 1 for testing and 1 for validation. So overall, i use 36 sequences for training, 4 for validation and 4 for testing 

## Task 1
I used RestNet50 as the backbone and only add a linear layer at the end to adjust to number of classes in our case
### Run the training
```python train.py --train --num_epoch 200 --batch_size 4 --lr 0,005```
### Run the evaluation to see how well the model performs on test set. 
The accuracy = accurate prediction / number of images in test set. I trained for 300 epochs and get the accuray of 99.62%

```python train.py --test```

### Test a specific image from the test set 
```python test.py --idx 1```

## Task 2

## References
https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
