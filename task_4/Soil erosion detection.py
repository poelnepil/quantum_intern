#!/usr/bin/env python
# coding: utf-8

# # Import the libraries

# In[221]:


import os

import rasterio
from rasterio.plot import reshape_as_image
import rasterio.mask
from rasterio.features import rasterize

import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, Point, Polygon
from shapely.ops import cascaded_union

import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from PIL import Image
from patchify import patchify

import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Dense, Dropout, Rescaling, concatenate

get_ipython().run_line_magic('matplotlib', 'inline')


# # Data analysis

# In[222]:


raster_path = r"C:\Users\vanzo\OneDrive\Рабочий стол\myownprojects\T36UXV_20200406T083559_TCI_10m.jp2"
with rasterio.open(raster_path, "r", driver="JP2OpenJPEG") as src:
    raster_img = src.read()
    raster_meta = src.meta
    raster_crs = src.crs


# In[223]:


raster_img.shape


# In[224]:


raster_meta


# In[225]:


raster_img = reshape_as_image(raster_img)


# In[226]:


plt.figure(figsize=(15,15))
plt.imshow(raster_img)


# In[227]:


train_df = gpd.read_file(r"C:\Users\vanzo\OneDrive\Рабочий стол\myownprojects\masks\Masks_T36UXV_20190427.shp")
print(len(train_df))
train_df.head(5)


# In[228]:


train_df.shape


# In[229]:


train_df.info()


# In[230]:


train_df.describe()


# In[231]:


train_df['geometry'][0].exterior.coords.xy


# http://projfinder.com/

# In[232]:


train_df = gpd.read_file(r"C:\Users\vanzo\OneDrive\Рабочий стол\myownprojects\masks\Masks_T36UXV_20190427.shp")
train_df = train_df[train_df.geometry.notnull()]
train_df.crs = {'init' :'epsg:4760'}
train_df = train_df.to_crs(raster_crs)


# In[233]:


train_df.head()


# In[234]:


src = rasterio.open(raster_path, 'r', driver="JP2OpenJPEG")
outfolder = r"C:\Users\vanzo\OneDrive\Рабочий стол\myownprojects\T36UXV_20200406T083559_TCI_10m.jp2"
failed = []
for num, row in train_df.iterrows():
    try:
        masked_image, out_transform = rasterio.mask.mask(src, [mapping(row['geometry'])], crop=True, nodata=0)
        img_image = reshape_as_image(masked_image)
        img_path = os.path.join(outfolder, str(row['Field_Id']) + '.png')
        img_image = cv2.cvtColor(img_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, img_image)
    except Exception as e:
        failed.append(num)
print("Rasterio failed to mask {} files".format(len(failed)))


# In[235]:


def poly_from_utm(polygon, transform):
    poly_pts = []
    for i in np.array(polygon.exterior.coords):
        poly_pts.append(~transform * tuple(i))
    new_poly = Polygon(poly_pts)
    return new_poly

poly_shp = []
im_size = (src.meta['height'], src.meta['width'])
for num, row in train_df.iterrows():
    if row['geometry'].geom_type == 'Polygon':
        poly = poly_from_utm(row['geometry'], src.meta['transform'])
        poly_shp.append(poly)
    else:
        for p in row['geometry']:
            poly = poly_from_utm(p, src.meta['transform'])
            poly_shp.append(poly)

mask = rasterize(shapes=poly_shp,
                 out_shape=im_size)

plt.figure(figsize=(15,15))
plt.imshow(mask)


# In[236]:


bin_mask_meta = src.meta.copy()
bin_mask_meta.update({'count': 1})
with rasterio.open(r"C:\Users\vanzo\OneDrive\Рабочий стол\mask.jp2", 'w', **bin_mask_meta) as dst:
    dst.write(mask * 255, 1)


# In[237]:


def load_images_and_patchify(directory_path, patch_size, color_change=False):
    instances = []
    with rasterio.open(directory_path, "r", driver="JP2OpenJPEG") as src:
        data = src.read()
        image = reshape_as_image(data)
        if color_change:
            patch_shape = (patch_size, patch_size, 3)
        else: 
            image = np.squeeze(image, axis=2)
            patch_shape = (patch_size, patch_size)
        size_x = (image.shape[1] // patch_size) * patch_size
        size_y = (image.shape[0] // patch_size) * patch_size
        image = Image.fromarray(image)
        image = np.array(image.crop((0, 0, size_x, size_y)))
        patch_img = patchify(image, patch_shape, step=patch_size)
        for j in range(patch_img.shape[0]):
            for k in range(patch_img.shape[1]):
                single_patch_img = patch_img[j, k]
                instances.append(np.squeeze(single_patch_img))
    return instances


# In[238]:


patch_size=64
tile_patches = load_images_and_patchify(directory_path=r"C:\Users\vanzo\OneDrive\Рабочий стол\myownprojects\T36UXV_20200406T083559_TCI_10m.jp2", patch_size=patch_size, color_change = True)
mask_patches = load_images_and_patchify(directory_path=r"C:\Users\vanzo\OneDrive\Рабочий стол\mask.jp2", patch_size=patch_size)


# In[239]:


len(tile_patches),len(mask_patches)


# In[240]:


import random as r
n = r.randrange(1,29241)
plt.imshow(mask_patches[17712])
plt.show()
plt.imshow(tile_patches[17712])
plt.show()


# In[241]:


threshold = 100
index_masks_with_erosion = []
percentage_of_erosion_list = [] 
for index in range(len(mask_patches)):
    image = mask_patches[index]
    total = sum(sum(row) for row in image) / 255
    if total > threshold:
        index_masks_with_erosion.append(index)
        percentage_of_erosion = total/(patch_size*patch_size)
        percentage_of_erosion_list.append (percentage_of_erosion)
quantity = len(index_masks_with_erosion)
print (f"quantity: {quantity}")
percentage_of_erosion_total = sum(percentage_of_erosion_list)/quantity
print (f"percentage of erosion: {percentage_of_erosion_total*100}%")


# In[242]:


tile_patches_erosion = list(map(lambda i: tile_patches[i], index_masks_with_erosion))
mask_patches_erosion = list(map(lambda i: mask_patches[i], index_masks_with_erosion))


# # Model 

# In[243]:


X,y = np.stack(tile_patches_erosion), np.stack(mask_patches_erosion) / 255


# In[244]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)


# In[245]:


def build_unet(img_shape):
    n_classes = 1
    
    inputs = Input(shape=img_shape)
    rescale = Rescaling(scale=1. / 255, input_shape=img_shape)(inputs)
    previous_block_activation = rescale  # Set aside residual

    contraction = {}
    # # Contraction path: Blocks 1 through 5 are identical apart from the feature depth
    for f in [16, 32, 64, 128]:
        x = Conv2D(f, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(previous_block_activation)
        x = Dropout(0.1)(x)
        x = Conv2D(f, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        contraction[f'conv{f}'] = x
        x = MaxPooling2D((2, 2))(x)
        previous_block_activation = x

    c5 = Conv2D(160, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(previous_block_activation)
    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(160, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    previous_block_activation = c5

    for f in reversed([16, 32, 64, 128]):
        x = Conv2DTranspose(f, (2, 2), strides=(2, 2), padding='same')(previous_block_activation)
        x = concatenate([x, contraction[f'conv{f}']])
        x = Conv2D(f, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(f, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        previous_block_activation = x

    outputs = Conv2D(filters=n_classes, kernel_size=(1, 1), activation="sigmoid")(previous_block_activation)

    return Model(inputs=inputs, outputs=outputs)


# In[246]:


def weighted_binary_crossentropy(weights):
    
    weights = tf.keras.backend.variable(weights)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        loss_pos = -weights[0] * y_true * tf.math.log(y_pred)
        loss_neg = -weights[1] * (1 - y_true) * tf.math.log(1 - y_pred)
        loss = tf.reduce_sum(loss_pos + loss_neg, axis=-1)

        return loss

    return loss


# In[247]:


patch_size=64
model = build_unet(img_shape=(patch_size, patch_size, 3))
model.summary()


# In[248]:


checkpoint = ModelCheckpoint(r"C:\Users\vanzo\OneDrive\Рабочий стол\myownprojects\model_per", monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")

early_stopping = EarlyStopping(monitor="val_loss", patience=6, verbose=1, mode="min")

callbacks_list = [checkpoint, early_stopping]

model.compile(optimizer="adam",loss=weighted_binary_crossentropy([9, 1]), metrics=["accuracy"])

model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), callbacks=callbacks_list, verbose=1)
model.save(r"C:\Users\vanzo\OneDrive\Рабочий стол\myownprojects\model")


# In[254]:


num = r.randrange(1,613)
index = index_masks_with_erosion[num]
predicted = model.predict(np.stack([tile_patches[index]]))
predicted_instance = predicted[0] * 255

plt.imshow(predicted_instance)
plt.show()
plt.imshow(mask_patches[index])
plt.show()
plt.imshow(tile_patches[index])
plt.show()

