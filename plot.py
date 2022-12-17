import numpy as np
import matplotlib.pyplot as plt
import torch

#Plotting accuracy
train_acc=np.load('/home/sbx5057/checkpoints/eyepacs/loss_accuracy_train.npy')
val_acc=np.load('/home/sbx5057/checkpoints/eyepacs/loss_accuracy_val.npy')
print(train_acc)
print(val_acc)

epochs = range(0,50)
plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Plotting Loss
train_loss=np.load('/home/sbx5057/checkpoints/eyepacs/loss_recorder_train.npy')
val_loss=np.load('/home/sbx5057/checkpoints/eyepacs/loss_recorder_val.npy')
print(train_loss)
print(val_loss)

epochs = range(0,50)
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
