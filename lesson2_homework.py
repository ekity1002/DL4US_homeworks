#!/usr/bin/env python
# coding: utf-8

# # Lesson2 畳み込みニューラルネットワーク (CNN)

# ## Homework

# 今Lessonで学んだことに工夫を加えて, CNNでより高精度なCIFAR10の分類器を実装してみましょう.
# 精度上位者はリーダーボードに載ります.

# ### 目標値

# Accuracy 90%

# ### ルール

# - ネットワークの形などは特に制限を設けません.
# - アンサンブル学習などを組み込んでもOKです.
# - **下のセルで指定されている`x_train`, `y_train`以外の学習データは使わないでください.**

# ### 評価について

# - テストデータ(`x_test`)に対する予測ラベルをcsvファイルで提出してください.
# - ファイル名は`submission.csv`としてください.
# - 予測ラベルの`y_test`に対する精度 (Accuracy) で評価します.
# - 毎日24時にテストデータの一部に対する精度でLeader Boardを更新します.
# - 最終的な評価はテストデータ全体に対する精度でおこないます.

# ### サンプルコード

# **次のセルで指定されている`x_train`, `y_train`のみを使って学習させてください.**

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


# In[3]:


print(x_train.shape, y_train.shape, x_test.shape)
print(y_train.sum(axis=0))


# In[4]:


# 水増し用ジェネレータ定義
def train_generator(x, y, batch_size):
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

def val_generator(x, y, batch_size):
    img_gen = ImageDataGenerator() # 検証用には処理を加えない!!
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


# In[5]:


def build_model():
    # 使うモデルに置き換える。
    drop_rate=0.3
    inp = Input(shape = (32,32,3))

    x = Conv2D(64,(3,3),padding = "SAME",activation= "relu", kernel_initializer='he_normal')(inp)
    x = Conv2D(64,(3,3),padding = "SAME",activation= "relu", kernel_initializer='he_normal')(x)
    x = Conv2D(64,(3,3),padding = "SAME",activation= "relu", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(drop_rate)(x)

    x = Conv2D(128,(3,3),padding = "SAME",activation= "relu", kernel_initializer='he_normal')(x)
    x = Conv2D(128,(3,3),padding = "SAME",activation= "relu", kernel_initializer='he_normal')(x)
    x = Conv2D(128,(3,3),padding = "SAME",activation= "relu", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(drop_rate)(x)

    x = Conv2D(256,(3,3),padding = "SAME",activation="relu",kernel_initializer='he_normal')(x)
    x = Conv2D(256,(3,3),padding = "SAME",activation= "relu", kernel_initializer='he_normal')(x)
    x = Conv2D(256,(3,3),padding = "SAME",activation= "relu", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(drop_rate)(x)

    x = Conv2D(512,(3,3),padding = "SAME",activation= "relu",kernel_initializer='he_normal')(x)
    x = Conv2D(512,(3,3),padding = "SAME",activation= "relu",kernel_initializer='he_normal')(x)
    x = Conv2D(512,(3,3),padding = "SAME",activation= "relu",kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(drop_rate)(x)
    
    x = Conv2D(1024,(3,3),padding = "SAME",activation= "relu",kernel_initializer='he_normal')(x)
    x = Conv2D(1024,(3,3),padding = "SAME",activation= "relu",kernel_initializer='he_normal')(x)
    x = Conv2D(1024,(3,3),padding = "SAME",activation= "relu",kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(2048,activation = "relu")(x)
    x = Dropout(drop_rate)(x)
    x = Dense(1024,activation = "relu")(x)
    x = Dropout(drop_rate)(x)
    out  = Dense(10,activation = "softmax")(x)

    return Model(inputs=inp, outputs=out)
    


# In[6]:


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


# In[7]:


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 100:
        lr *= 0.2
    elif epoch > 140:
        lr *= 0.5
    print('Learning rate: ', lr)
    return lr


# In[ ]:


# cv
def run_cv(train, test, target, params={}):
    N = 5
    kf = StratifiedKFold(n_splits=N, random_state=SEED)
    fold_splits = kf.split(train, target.argmax(axis=1))
    tr_scores = []
    val_scores = []
    results = np.zeros((test.shape[0], N))
    i = 0
    
    for tr_idx, val_idx in fold_splits:
        print(f'Start fold {i}/{N}')
        tr_X, val_X = train[tr_idx, :], train[val_idx, :]
        tr_y, val_y = target[tr_idx, :], target[val_idx, :]
        params['modelpath'] = f'/root/userspace/lesson2/second_submission_cv{i}_model.hdf5'

        tr_acc, val_acc, test_pred = run_model(tr_X, tr_y, val_X, val_y, test, params)
        tr_scores.append(tr_acc)
        val_scores.append(val_acc)
        results[:, i-1] = test_pred
        i+=1
        
    print('mean acc: ', sum(val_scores)/len(val_scores))
    return results

# モデル予測実行
def run_model(tr_X, tr_y, val_X, val_y, test, params, gen_func=None):
    print('Train model')
    batch_size=params['batch_size']
    epochs = params['epochs']
    modelpath = params['modelpath']

    model = build_model()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    
    callbacks = [
        EarlyStopping(monitor='val_acc', patience=20),
        ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=10),
        ModelCheckpoint(filepath=modelpath, monitor='val_acc', save_best_only=True)
    ]
    model.fit_generator(my_generator(tr_X, tr_y, batch_size),
                        steps_per_epoch=len(tr_X)//batch_size,
                        epochs=epochs,
                        validation_data=my_generator(val_X, val_y, batch_size),
                        validation_steps=len(val_X)//batch_size,
                        callbacks=callbacks,
                        verbose=2)
    print('Pred 1/2')
    tr_loss, tr_acc = model.evaluate(tr_X, tr_y)
    val_loss, val_acc = model.evaluate(val_X, val_y)
    print(f'[Train] acc:{tr_acc}  loss:{tr_loss}')
    print(f'[Val]   acc:{val_acc} loss:{val_loss}')
    print('Pred 2/2')
    #pred_test = np.argmax(model.predict(test), axis=1)

    # try tta
    tta_epochs = 50
    pred_test = np.argmax(tta(model, x_test.shape[0], tta_generator(x_test, batch_size=1000), batch_size=1000, epochs=tta_epochs), axis=1)
    return tr_acc, val_acc, pred_test

params = {'batch_size':1000,
          'epochs':1000,}
results = run_cv(x_train, x_test, y_train, params)


# In[8]:


model = load_model('/root/userspace/lesson2/second_submission_cv1_model.hdf5')
# try tta
tta_epochs = 50
y_pred = np.argmax(tta(model, x_test.shape[0], tta_generator(x_test, batch_size=1000), batch_size=1000, epochs=tta_epochs), axis=1)


# In[9]:


submission = pd.DataFrame(results, dtype=int)
submission = submission.apply(lambda x: np.argmax(x.value_counts()), axis=1)
submission.head(20)


# In[ ]:


submission = pd.Series(y_pred, name='label')
submission.to_csv('/root/userspace/last_submission.csv', header=True, index_label='id')
