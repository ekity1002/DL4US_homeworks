#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import tensorflow.keras as keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Input, Activation, add, Add, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, StratifiedKFold

SEED = 2019


# In[2]:


def load_cifar10():
    
    # 学習データ
    x_train = np.load('/root/userspace/public/lesson2/data/x_train.npy')
    y_train = np.load('/root/userspace/public/lesson2/data/y_train.npy')

    # テストデータ
    x_test = np.load('/root/userspace/public/lesson2/data/x_test.npy')
    
    x_train = x_train / 255.
    x_test = x_test / 255.

    # 平均値を引く
    subtract_pixel_mean = True
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
        
    y_train = np.eye(10)[y_train]    
    return (x_train, x_test, y_train)

x_train, x_test, y_train = load_cifar10()


# In[13]:


# 水増し用ジェネレータ定義
def my_generator(x, y, batch_size):
    gen_params = {
        'rotation_range':20,
        'horizontal_flip':True,
        'height_shift_range':0.2,
        'width_shift_range':0.2,
        'zoom_range':0.2,
        'channel_shift_range':0.2
    }
    img_gen = ImageDataGenerator(**gen_params)
    for x_batch, y_batch in img_gen.flow(x,y,batch_size=batch_size):
        yield x_batch, y_batch
        
def tta(model, test_size, generator, batch_size ,epochs=10):
    # test_time_augmentation 予測画像を水増しし、多数決を取る。
    # batch_sizeは，test_sizeの約数でないといけない．
    pred = np.zeros(shape = (test_size,10), dtype=float)
    step_per_epoch = test_size // batch_size #1バッジに使うテストデータの量
    for epoch in range(epochs):
        for step in range(step_per_epoch):
            sta = batch_size * step #バッジの先頭のデータのテストデータにおけるインデックス ?
            end = sta + batch_size  # バッジ終了のテストデータのインデックス
            tmp_x = generator.__next__() #予測画像をbatch_size だけ生成
            #print("tmpx : ", tmp_x.shape)
            pred[sta:end] += model.predict(tmp_x)

    return pred / epochs


def tta_generator(x_test, batch_size):
    """tta用 予測画像水増し用ジェネレータ"""
    return ImageDataGenerator(rotation_range = 20 , horizontal_flip = True,height_shift_range = 0.2,                                 width_shift_range = 0.2,zoom_range = 0.2,channel_shift_range = 0.2                                  ).flow(x_test, batch_size = batch_size,shuffle = False)


# In[4]:


def plot_acc_and_loss(history):
    """plot history"""
    fig, ax = plt.subplots(1,2, figsize=(15, 5))
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']    
    epochs=range(1, len(acc)+1)    
    ax[0].plot(epochs, acc, label='Train')
    ax[0].plot(epochs, val_acc, label='Val')
    ax[0].legend()
    ax[0].set_title('Accuracy')
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs=range(1, len(loss)+1)
    ax[1].plot(epochs, loss, label='Train')
    ax[1].plot(epochs, val_loss, label='Val')
    ax[1].legend()
    ax[1].set_title('Loss')
    plt.show()


# In[16]:


def build_model():
    # 使うモデルに置き換える。
    drop_rate=0.3
    inp = Input(shape = (32,32,3))

    x = Conv2D(64,(3,3),padding='same',activation="relu", kernel_initializer='he_normal')(inp)
    x = Conv2D(64,(3,3),padding='same',activation="relu", kernel_initializer='he_normal')(x)
    x = Conv2D(64,(3,3),padding='same',activation="relu", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(drop_rate)(x)

    x = Conv2D(128,(3,3),padding='same',activation="relu", kernel_initializer='he_normal')(x)
    x = Conv2D(128,(3,3),padding='same',activation="relu", kernel_initializer='he_normal')(x)
    x = Conv2D(128,(3,3),padding='same',activation="relu", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(drop_rate)(x)

    x = Conv2D(256,(3,3),padding='same',activation="relu",kernel_initializer='he_normal')(x)
    x = Conv2D(256,(3,3),padding='same',activation="relu", kernel_initializer='he_normal')(x)
    x = Conv2D(256,(3,3),padding='same',activation="relu", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(drop_rate)(x)

    x = Conv2D(512,(3,3),padding='same',activation="relu",kernel_initializer='he_normal')(x)
    x = Conv2D(512,(3,3),padding='same',activation="relu",kernel_initializer='he_normal')(x)
    x = Conv2D(512,(3,3),padding='same',activation="relu",kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(drop_rate)(x)
    
    x = Conv2D(1024,(3,3),padding="same",activation="relu",kernel_initializer='he_normal')(x)
    x = Conv2D(1024,(3,3),padding='same',activation="relu",kernel_initializer='he_normal')(x)
    x = Conv2D(1024,(3,3),padding='same',activation="relu",kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024,activation="relu", kernel_initializer='he_normal')(x)
    x = Dropout(drop_rate)(x)
    x = Dense(516,activation="relu", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dense(1024,activation="relu", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    out  = Dense(10,activation="softmax")(x)

    return Model(inputs=inp, outputs=out)


# In[17]:


get_ipython().run_cell_magic('time', '', "model = build_model()\nmodel.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])\n\nX_trn, X_val, y_trn, y_val = train_test_split(x_train, y_train, random_state=SEED, stratify=y_train)\nbatch_size = 800\nepochs = 1000\nsteps_per_epoch = X_trn.shape[0] // batch_size\nvalidation_steps = X_val.shape[0] // batch_size\ncallbacks = [EarlyStopping(patience=30),\n             ReduceLROnPlateau(patience=15, factor=0.2),\n             #LearningRateScheduler(lr_schedule),\n             ModelCheckpoint(filepath='/root/userspace/lesson2/expmodel_1.hdf5', monitor='val_acc', save_best_only=True)\n            ]\n\nval_gen = ImageDataGenerator().flow(X_val, y_val, batch_size)\nhistory = model.fit_generator(my_generator(X_trn, y_trn, batch_size),\n                    steps_per_epoch=len(X_trn)//batch_size,\n                    epochs=epochs,\n                    validation_data=val_gen,\n                    validation_steps=len(X_val)//batch_size,\n                    callbacks=callbacks,\n                    verbose=2)\n#model.fit(x=x_train, y=y_train, batch_size=700, epochs=150, validation_split=0.1)\n\ny_pred = model.predict(x_test)\ny_pred = np.argmax(y_pred, 1)\nsubmission = pd.Series(y_pred, name='label')\nsubmission.to_csv('/root/userspace/exp_submission.csv', header=True, index_label='id')")


# In[18]:


plot_acc_and_loss(history)


# In[19]:


get_ipython().run_cell_magic('time', '', '# try tta\ntta_epochs = 50\ntta_pred = tta(model, x_test.shape[0], tta_generator(x_test, batch_size=1000), batch_size=1000, epochs=tta_epochs)\nprint(tta_pred.shape)\nprint(tta_pred)')


# In[20]:


tta_pred = np.argmax(tta_pred, axis=1)
submission = pd.Series(tta_pred, name='label')
submission.to_csv('/root/userspace/exptta_submission.csv', header=True, index_label='id') 



