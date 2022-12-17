#Reference Page - https://www.kaggle.com/code/umangtri/diabetic-retinopathy-version-2#Modelling-(base-Models)
import numpy as np 
import pandas as pd 
import os
import cv2
import random
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import gaussian, convolve2d
import seaborn as sns

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D, Dropout,Activation, GlobalAveragePooling2D, BatchNormalization, GlobalMaxPooling2D
from tensorflow.keras.applications import DenseNet121, ResNet50, InceptionV3, Xception, VGG16,ResNet101
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend
from tensorflow.keras.regularizers import l2, l1
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score


#Class Labels    
classes = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

#Image Path    
dir_path = '/home/sbx5057/Documents/COMP597/eyepacs_preprocess/eyepacs_preprocess'
os.listdir(dir_path)
#Image Label Path
df_temp = pd.read_csv("/home/sbx5057/Documents/COMP597/trainLabels.csv")
print(len(df_temp), df_temp)
print(df_temp['level'].value_counts())

#Relating the given class codes to label names
class_code = {0: "No_DR",
              1: "Mild", 
              2: "Moderate",
              3: "Severe",
              4: "Proliferate_DR"}
df_temp.rename(columns={"image": "id_code", "level": "diagnosis"}, inplace=True)

def mapping_temp(df, root=dir_path):
    class_code = {0: "No_DR",
                  1: "Mild", 
                  2: "Moderate",
                  3: "Severe",
                  4: "Proliferate_DR"}
    df['label'] = list(map(class_code.get, df['diagnosis']))
    df['path'] = [i[1]['label']+'/'+i[1]['id_code']+'.jpeg' for i in df.iterrows()]
    return df
mapping_temp(df_temp)

#Wiener filter for noise reduction
def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = np.fft.fft2(dummy)
    kernel = np.fft.fft2(kernel, s = img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(np.fft.ifft2(dummy))
    return dummy

def gaussian_kernel(kernel_size = 3):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h

def isbright(image, dim=227, thresh=0.4):
    # Resize image to 10x10
    image = cv2.resize(image, (dim, dim))
    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L/np.max(L)
    # Return True if mean is greater than thresh else False
    return np.mean(L) > thresh


#Function for image preprocessing
def image_preprocessing(img):
    img = img.astype(np.uint8) #Read image
    b, g, r = cv2.split(img)  #Splitting the image into the 3 channels

    # Applying wiener filter to reduce noise
    wiener = wiener_filter(g, gaussian_kernel(3), 10)
    g = cv2.addWeighted(g, 1.5, wiener.astype("uint8"), -0.5, 0)

    #Intensify image using CLAHE
    clh = cv2.createCLAHE(clipLimit=3.0)
    g = clh.apply(g.astype('uint8'))
    
    #Merging the 3 channels
    merged_bgr_green_fused = cv2.merge((b, g, r))
 
    
    # Checking for intensity of image
    if isbright(merged_bgr_green_fused)==False:
        output_gaussian = cv2.GaussianBlur(img, (5, 5), 0)
        merged_bgr_green_fused = cv2.addWeighted(merged_bgr_green_fused, 1.5, output_gaussian, -0.5, 0)
    return merged_bgr_green_fused.astype("float64")
   
#Displaying one image from the dataset after preprocessing    
p = "/home/sbx5057/Documents/COMP597/eyepacs_preprocess/eyepacs_preprocess/35667_left.jpeg"
img = mpimg.imread(p)
pro = image_preprocessing(img)
filename = os.path.basename(p)
plt.imshow(pro.astype("uint8"), cmap="gray");
#plt.show()


random_img_path = [dir_path+'/'+img for img in random.sample(os.listdir(dir_path), 50)]
random_img_path

#Displaying 50 random images
plt.figure(figsize=(20, 15))
plt.suptitle("Image Dataset for CLAHE Processed Images", fontsize=20)
for i in range(1, 51):
    plt.subplot(5, 10, i)
    img = mpimg.imread(random_img_path[i-1])
    img_pro = image_preprocessing(img)
    plt.imshow(img_pro.astype("uint8"), cmap="gray", aspect="auto")
    plt.axis(False);
plt.show()
    
for i in range(5):
  try:
    os.mkdir('./'+class_code[i])
  except FileExistsError:
    pass
    
import os
import shutil
res = [[i[1][2], i[1][3]] for i in df_temp.iterrows()]

for i in res:
    #print("Hey");
    des = './'+i[0]+'/'
    src = dir_path+'/'+i[1].split('/')[1]
    print(src);
    shutil.copy(src, des)

#Splitting into training and testing datasets
train_df_temp = {}
test_df_temp = {}
for i in range(5):
    df = df_temp[df_temp['diagnosis']==i]['id_code'].to_list()
    random.seed(42)
    x = random.sample(df, int(0.8*len(df)))
    for j in x:
        train_df_temp[j] = i
    for j in df:
        if j not in train_df_temp.keys():
            test_df_temp[j] = i
train_df_temp = pd.DataFrame(train_df_temp.items(), columns=['id_code', 'diagnosis']).sample(frac=1, random_state=42)
test_df_temp = pd.DataFrame(test_df_temp.items(), columns=['id_code', 'diagnosis']).sample(frac=1, random_state=42)
print(train_df_temp)   

# Taking only 20% of the dataset
train_df_temp_10_per = {}
test_df_temp_10_per = {}
for i in range(5):
    df = train_df_temp[train_df_temp['diagnosis']==i]['id_code'].to_list()
    random.seed(42)
    x = random.sample(df, int(0.2*len(df)))
    for j in x:
        train_df_temp_10_per[j] = i
    for j in df:
        if j not in train_df_temp_10_per.keys():
            test_df_temp_10_per[j] = i
train_df_temp_10_per = pd.DataFrame(train_df_temp_10_per.items(), columns=['id_code', 'diagnosis']).sample(frac=1, random_state=42)
test_df_temp_10_per = pd.DataFrame(test_df_temp_10_per.items(), columns=['id_code', 'diagnosis']).sample(frac=0.05, random_state=4)
print(train_df_temp_10_per )
print(len(test_df_temp_10_per))

mapping_temp(train_df_temp, root='.'), mapping_temp(test_df_temp, root='.')
mapping_temp(train_df_temp_10_per, root='.'), mapping_temp(test_df_temp_10_per, root='.')


#Initializing for model evaluation
IMG_SHAPE = (224, 224)
N_SPLIT = 3
EPOCHS = 5

#Euclidean Distance calculation
def euclideanDist(img1, img2):
    return backend.sqrt(backend.sum((img1-img2)**2))

#Metrics Evaluation
def metrics(y_true, y_pred):
    print(classification_report(y_true, y_pred, target_names=classes))
    acc = accuracy_score(y_true, y_pred)
    res = []
    for l in [0,1,2,3,4]:
        prec,recall,_,_ = precision_recall_fscore_support(np.array(y_true)==l,
                                                          np.array(y_pred)==l,
                                                          pos_label=True,
                                                          average=None)
        res.append([classes[l],recall[0],recall[1]])
    df_res = pd.DataFrame(res,columns = ['class','sensitivity','specificity'])
    return df_res, acc

# Function to perform k-fold = 3-fold validation on test model
def validation_k_fold_temp(model_test, k=3, epochs=EPOCHS, n_splits=N_SPLIT, lr=0.001, class_weights=None): 
    kfold = StratifiedKFold(n_splits=N_SPLIT,shuffle=True,random_state=42)
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                    preprocessing_function = image_preprocessing)
    validation_datagen = ImageDataGenerator(rescale = 1./255,
                                            preprocessing_function = image_preprocessing)

    train_y = train_df_temp_10_per['label']
    train_x = train_df_temp_10_per['path']

    # Variable for keeping the count of the splits we're executing
    j = 0
    es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    histories, acc = [], []
    for train_idx, val_idx in list(kfold.split(train_x,train_y)):
        x_train_df = train_df_temp_10_per.iloc[train_idx]
        x_valid_df = train_df_temp_10_per.iloc[val_idx]
        j+=1
        train_data = train_datagen.flow_from_dataframe(dataframe=x_train_df, 
                                                       directory='./',
                                                       x_col='path',
                                                       y_col='label',
                                                       class_mode="categorical",
                                                       target_size=IMG_SHAPE)

        valid_data = validation_datagen.flow_from_dataframe(dataframe=x_valid_df, 
                                                           directory='./',
                                                           x_col='path',
                                                           y_col='label',
                                                           class_mode="categorical",
                                                           target_size=IMG_SHAPE)
        
        # Initializing the early stopping callback
        es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        
        # Compile the model
        model_test.compile(loss='categorical_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          metrics=[euclideanDist, 'accuracy'])
        
        history = model_test.fit_generator(train_data,
                                           validation_data=valid_data,
                                           epochs=epochs,
                                           validation_steps=len(valid_data),
                                           callbacks=[es])
        histories.append(history.history)
        
        test_datagen = ImageDataGenerator(rescale = 1./255,
                                          preprocessing_function = image_preprocessing)
        test_data = test_datagen.flow_from_dataframe(dataframe=test_df_temp_10_per, 
                                                     directory='./',
                                                     x_col='path',
                                                     y_col='label',
                                                     class_mode="categorical",
                                                     target_size=IMG_SHAPE)
        
        # Evaluate the model
        predictions = model_test.predict_generator(test_data, verbose=1)
        y_preds = np.argmax(predictions, axis=1)
        true_classes = test_data.classes
        
        # evaluate test performance
        print("***Performance on Test data***")    
        df_res, testAcc = metrics(true_classes, y_preds)
        acc.extend([testAcc, df_res])
    return [histories, acc]
    
def print_met(model):
    # Evaluation metrics for model
    for i in range(0, len(model), 2):
        print(f"Accuracy: {model[i]}")
        print(model[i+1])
        print("\t\t------------------\n")
        
        
# Function to plot the performance metrics
def plot_result(hist):
    plt.figure(figsize=(20, 10));
    plt.suptitle(f"Performance Metrics", fontsize=20)
    
    c=1
    for i in range(N_SPLIT):
        # Actual and validation losses
        plt.subplot(3, 3, c);
        plt.plot(hist[i]['loss'], label='train')
        plt.plot(hist[i]['val_loss'], label='validation')
        plt.title('Train and val loss curve')
        plt.legend()

        # Actual and validation accuracy
        plt.subplot(3, 3, c+1);
        plt.plot(hist[i]['accuracy'], label='train')
        plt.plot(hist[i]['val_accuracy'], label='validation')
        plt.title('Train and val accuracy curve')
        plt.legend()

        # Actual and validation euclidean distance
        plt.subplot(3, 3, c+2);
        plt.plot(hist[i]['euclideanDist'], label='train')
        plt.plot(hist[i]['val_euclideanDist'], label='validation')
        plt.title('Train and val euclidean distance curve')
        plt.legend()
        c+=3
    plt.tight_layout()
    plt.show()
    
# View random images in the dataset
def view_random_images(root_dir, classes=classes):
    class_paths = [root_dir + "/" + image_class for image_class in classes]
    # print(class_paths)
    images_path = []
    labels = []
    for i in range(len(class_paths)):
        random_images = random.sample(os.listdir(class_paths[i]), 10)
        random_images_path = [class_paths[i]+'/'+img for img in random_images]
        for j in random_images_path:
            images_path.append(j)
            labels.append(classes[i])
    images_path

    plt.figure(figsize=(17, 10))
    plt.suptitle("Image Dataset", fontsize=20)

    for i in range(1, 51):
        plt.subplot(5, 10, i)
        img = mpimg.imread(images_path[i-1])
        plt.imshow(img, aspect="auto")
        plt.title(labels[i-1])
        plt.axis(False);
    #plt.show()
        
# Observing the images
view_random_images(root_dir='./')     

#ALEXNET Architecture 
model_alexnet = tf.keras.Sequential([
    Conv2D(input_shape=IMG_SHAPE+(3,), filters=96,kernel_size=11,strides=4,activation='relu'),
    MaxPool2D(pool_size=3,strides=2),
    Conv2D(filters=256,kernel_size=5,strides=1,padding='valid',activation='relu'),
    MaxPool2D(pool_size=3,strides=2),
    Conv2D(filters=384,kernel_size=3,strides=1,padding='same',activation='relu'),
    Conv2D(filters=384,kernel_size=3,strides=1,padding='same',activation='relu'),
    Conv2D(filters=256,kernel_size=3,strides=1,padding='same',activation='relu'),
    MaxPool2D(pool_size=3,strides=2),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Flatten(),
    Dense(len(classes), activation='softmax')
], name="model_AlexNet")

model_alexnet.summary()

layer = model_alexnet.layers
for i in range(len(layer)):
    print(i, layer[i].trainable, layer[i].name)
    
model_alexnet_history, model_alexnet_result = validation_k_fold_temp(model_alexnet, class_weights=None)
print_met(model_alexnet_result)

# Performance metrics
plot_result(model_alexnet_history)


#DENSENET Architecture
model_densenet=DenseNet121(weights='imagenet',include_top=False, input_shape=(224, 224, 3)) 
x=model_densenet.output
x= GlobalMaxPooling2D()(x)
x= BatchNormalization()(x)
x= Dense(256, activation='relu')(x)
x= Dropout(0.5)(x)
output=Dense(len(classes),activation='softmax')(x)
model_denseNet=tf.keras.Model(inputs=model_densenet.input,outputs=output)


model_denseNet.summary()
layers = model_denseNet.layers
for i in range(len(layers)):
    print(i, layers[i].trainable, layers[i].name)
    
# Freezing the base model
for layer in model_denseNet.layers[:-5]:
    layer.trainable=False

model_denseNet_history, model_denseNet_result = validation_k_fold_temp(model_denseNet, class_weights=None)

# Evaluation metrics for denseNet model
print_met(model_denseNet_result)

# Visualizing the evaluation metrics
plot_result(model_denseNet_history)


#RESNET50 Architecture
model_resnet50=ResNet50(weights='imagenet',include_top=False, input_shape=(224, 224, 3)) 
x=model_resnet50.output
x= GlobalMaxPooling2D()(x)
x= BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x= Dropout(0.5)(x)
output=Dense(len(classes),activation='softmax')(x)
model_resNet=tf.keras.Model(inputs=model_resnet50.input,outputs=output)
l = model_resNet.layers

for i in range(len(l)):
    print(i, l[i].trainable, l[i].name)

#Print model summary    
model_resNet.summary()

#Freeze layers
for layer in model_resNet.layers[:-5]:
    layer.trainable=False
    
model_resNet_history, model_resNet_result = validation_k_fold_temp(model_resNet, class_weights=None)

#Print Metric Evaluation
print_met(model_resNet_result)

# Visualizing the evaluation metrics
plot_result(model_resNet_history)



#RESNET101 Architecture
model_resnet101=ResNet101(weights='imagenet',include_top=False, input_shape=(224, 224, 3)) 
x=model_resnet101.output
x=Conv2D(32,(3,3),strides=(2,2))(x)
x=Conv2D(32,(3,3),strides=(2,2))(x)
x= GlobalMaxPooling2D()(x)
output=Dense(2048,activation='softmax')(x)
model_resNet=tf.keras.Model(inputs=model_resnet101.input,outputs=output)

l = model_resNet.layers
for i in range(len(l)):
    print(i, l[i].trainable, l[i].name)
    
model_resNet.summary()

for layer in model_resNet.layers[:-5]:
    layer.trainable=False
    
model_resNet_history, model_resNet_result = validation_k_fold_temp(model_resNet, class_weights=None)

print_met(model_resNet_result)

# Visualizing the evaluation metrics
plot_result(model_resNet_history)


#INCEPTIONV3
model_inception=InceptionV3(weights='imagenet',include_top=False, input_shape=(224, 224, 3)) 
x=model_inception.output
x= GlobalAveragePooling2D()(x)
x= BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x= Dropout(0.5)(x)
output=Dense(len(classes),activation='softmax')(x)
model_inceptionV3=tf.keras.Model(inputs=model_inception.input,outputs=output)


model_inceptionV3.summary()

# Freezing the base model
for layer in model_inceptionV3.layers[:-5]:
    layer.trainable=False
    
model_inception_history, model_inception_result = validation_k_fold_temp(model_inceptionV3,class_weights=None)

# Evaluation metrics for InceptionV3
print_met(model_inception_result)

# Visualizing the evaluation metrics
plot_result(model_inception_history)


#VGG16 Architecture
model_vgg=VGG16(weights='imagenet',include_top=False, input_shape=(224, 224, 3)) 
x=model_vgg.output
x= GlobalAveragePooling2D()(x)
x= Dense(256, activation='relu')(x)
x= Dropout(0.5)(x)
output=Dense(len(classes),activation='softmax')(x) #FC-layer
model_vgg=tf.keras.Model(inputs=model_vgg.input,outputs=output)

# Summary
model_vgg.summary()

# Freezing the base model
for layer in model_vgg.layers[:-5]:
    layer.trainable=False
    
# Evaluation metrics
print_met(model_vgg_result)

# Visualizing the evaluation metrics
plot_result(model_vgg_history)
