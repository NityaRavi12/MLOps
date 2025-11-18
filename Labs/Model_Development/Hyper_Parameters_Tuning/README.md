# MNIST Model Development and Hyperparameter Tuning  
### MLOps – Model Development Lab

This repository contains my implementation of the Model Development Lab for the MLOps course.  
The goal of this lab is to build, train, tune, and evaluate a neural network model using Keras and Keras Tuner, while making meaningful modifications to the original notebook provided by the professor.

---

## Objectives

- Load and preprocess the MNIST dataset  
- Build a custom neural network model  
- Perform hyperparameter tuning  
- Train and evaluate the best model configuration  
- Add model visualizations (training curves, confusion matrix)  
- Save the final model for reuse  
- Make independent modifications so the notebook is not identical to the professor's version  

---

# Modifications I Made 

1. Used a modified model architecture  
   - Added Batch Normalization  
   - Added Dropout  
   - Used GELU activation instead of ReLU  
   - Optionally replaced CNN with MLP-only model to reduce training time

2. Changed the hyperparameter tuning approach  
   - Replaced Hyperband with RandomSearch for faster performance  
   - Set a fixed `max_trials` for predictable runtime  
   - Tuned number of neurons and learning rate

3. Added Data Augmentation  
   - Random rotation  
   - Random zoom  
   - Random width/height shifts  

4. Reorganized and cleaned the code structure  
   - Added markdown descriptions for each cell  
   - Improved readability  
   - Clear separation of model building, tuning, training, and evaluation

5. Added new evaluation visualizations  
   - Accuracy and loss learning curves  
   - Confusion matrix using scikit-learn  

6. Saved the final trained model  
   - Using Keras `.h5` or SavedModel format  

---

# Dataset

The MNIST dataset (handwritten digits) is used for this lab.

- 60,000 training images  
- 10,000 test images  
- Each image is 28×28 grayscale  

The dataset is loaded through Keras:

```python
from tensorflow.keras.datasets import mnist
