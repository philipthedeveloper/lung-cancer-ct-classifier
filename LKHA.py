#!/usr/bin/env python
# coding: utf-8

# In[7]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.metrics import precision_score, recall_score, mean_squared_error, mean_absolute_error, confusion_matrix
from sklearn.metrics import precision_score, recall_score, mean_squared_error, mean_absolute_error, confusion_matrix
import seaborn as sns
import itertools


# In[8]:


# Define directories
# train_dir = 'Documents/LAWAL KHADIJAH/train'
# valid_dir = 'Documents/LAWAL KHADIJAH/valid'
# test_dir = 'Documents/LAWAL KHADIJAH/test'
train_dir = './train'
valid_dir = './valid'
test_dir = './test'


# In[9]:


# ImageDataGenerator for data preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


# In[10]:


# Load datasets
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
valid_generator = valid_datagen.flow_from_directory(valid_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')


# In[11]:


def plot_images(generator, title):
    x, y = next(generator)
    class_indices = {v: k for k, v in generator.class_indices.items()}
    plt.figure(figsize=(12, 8))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(x[i])
        plt.title({class_indices[np.argmax(y[i])]}, fontsize = 8)
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# Plot samples from each dataset
plot_images(train_generator, "Training samples")
plot_images(valid_generator, "Validation samples")
plot_images(test_generator, "Test samples")


# In[12]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.summary()


# In[13]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, epochs=30, validation_data=valid_generator)


# In[14]:


# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()


# In[15]:


train_loss, train_acc = model.evaluate(train_generator)
print(f'Training loss: {train_loss}')
print(f'Training accuracy: {train_acc}')

y_true = train_generator.classes
y_pred = np.argmax(model.predict(train_generator), axis=-1)

precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'RMSE: {rmse}')


# In[16]:


val_loss, val_acc = model.evaluate(valid_generator)
print(f'Validation loss: {val_loss}')
print(f'Validation accuracy: {val_acc}')

y_true = valid_generator.classes
y_pred = np.argmax(model.predict(valid_generator), axis=-1)

precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'RMSE: {rmse}')


# In[17]:


test_loss, test_acc = model.evaluate(test_generator)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_acc}')

y_true = test_generator.classes
y_pred = np.argmax(model.predict(test_generator), axis=-1)

precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'RMSE: {rmse}')


# In[18]:


cm = confusion_matrix(y_true, y_pred)
cm_plot_labels = list(valid_generator.class_indices.keys())

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(20, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

plot_confusion_matrix(cm, cm_plot_labels)
plt.show()


# In[19]:


def plot_predictions(generator, model, num_images=8):
    x, y = next(generator)
    predictions = model.predict(x)
    class_indices = {v: k for k, v in generator.class_indices.items()}
    plt.figure(figsize=(15, 12))
    for i in range(num_images):
        plt.subplot(4, 4, i+1)
        plt.imshow(x[i])
        true_label = class_indices[np.argmax(y[i])]
        pred_label = class_indices[np.argmax(predictions[i])]
        plt.title(f'True: {true_label}, \nPred: {pred_label}', fontsize = 6.5)
        plt.axis('off')
    plt.show()

plot_predictions(valid_generator, model)


# In[20]:


def plot_predictions(generator, model, num_images=8):
    x, y = next(generator)
    predictions = model.predict(x)
    class_indices = {v: k for k, v in generator.class_indices.items()}
    plt.figure(figsize=(15, 12))
    for i in range(num_images):
        plt.subplot(4, 4, i+1)
        plt.imshow(x[i])
        true_label = class_indices[np.argmax(y[i])]
        pred_label = class_indices[np.argmax(predictions[i])]
        plt.title(f'True: {true_label}, \nPred: {pred_label}', fontsize = 6.5)
        plt.axis('off')
    plt.show()

plot_predictions(test_generator, model)


# In[21]:


# For multi-class, sensitivity per class
from sklearn.metrics import recall_score
val_loss, val_acc = model.evaluate(valid_generator)
print(f'Validation loss: {val_loss}')
print(f'Validation accuracy: {val_acc}')

y_true = valid_generator.classes
y_pred = np.argmax(model.predict(valid_generator), axis=-1)

# Calculate sensitivity for each class
sensitivities = recall_score(y_true, y_pred, average=None)
print(f"Sensitivities: {sensitivities}")

# If you want the average sensitivity
average_sensitivity = recall_score(y_true, y_pred, average='macro')  # or 'weighted'
print(f"Average Sensitivity: {average_sensitivity}")


# In[22]:


y_true = train_generator.classes
y_pred = np.argmax(model.predict(train_generator), axis=-1)

# Calculate sensitivity for each class
sensitivities = recall_score(y_true, y_pred, average=None)
print(f"Sensitivities: {sensitivities}")

# If you want the average sensitivity
average_sensitivity = recall_score(y_true, y_pred, average='macro')  # or 'weighted'
print(f"Average Sensitivity: {average_sensitivity}")


# In[23]:


y_true = test_generator.classes
y_pred = np.argmax(model.predict(test_generator), axis=-1)

# Calculate sensitivity for each class
sensitivities = recall_score(y_true, y_pred, average=None)
print(f"Sensitivities: {sensitivities}")

# If you want the average sensitivity
average_sensitivity = recall_score(y_true, y_pred, average='macro')  # or 'weighted'
print(f"Average Sensitivity: {average_sensitivity}")


# In[ ]:
model.save('model.h5')



