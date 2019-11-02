#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Lesson1 手書き文字認識をしよう（ニューラルネットワーク入門）


# ## Homework
# 今Lessonで学んだMLPを用いて、FashionMNISTの高精度な分類器を実装してみましょう。 
# 
# モデルのレイヤーを変更してみるなどして精度の向上にチャンレンジして下さい。
# 精度上位者はリーダーボードに載ります。

# ## 目標値
# 
# Accuracy 90%

# ### ルール
# - 訓練データは`x_train`、 `y_train`、テストデータは`x_test`で与えられます。
# - 予測ラベルは **one_hot表現ではなく0~9のクラスラベル** で表してください。
# - 下のセルで指定されているx_train、y_train以外の学習データは使わないでください。
# - 次のレッスンでCNNを学習するので今回の宿題での利用は控えて下さい。
# 
# ### 評価について
# 
# - テストデータ(x_test)に対する予測ラベルをcsvファイルで提出してください。
# - ファイル名はsubmission.csvとしてください。
# - 予測ラベルのy_testに対する正解率(accuracy)で評価します。
# - 評価はテストデータに対する正解率でおこないます。

# ### サンプルコード
# 
# - **次のセルで指定されているx_train, y_trainのみを使って学習させてください。**
# - `submission.csv`の出力場所は適宜変更して下さい。

# In[2]:


import numpy as np
import pandas as pd
import os
import random
#import keras.backend as K
import tensorflow as tf

SEED=10000
def fix_seed():
    np.random.seed(seed=SEED)

    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(0)

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )

    tf.set_random_seed(0)
    # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    # K.set_session(sess)
#fix_seed()


# In[3]:


def load_mnist():

    # 学習データ
    x_train = np.load('/root/userspace/public/lesson1/data/x_train.npy')
    y_train = np.load('/root/userspace/public/lesson1/data/y_train.npy')
    
    # テストデータ
    x_test = np.load('/root/userspace/public/lesson1/data/x_test.npy')

    x_train = x_train.reshape(-1, 784).astype('float32') / 255
    x_test = x_test.reshape(-1, 784).astype('float32') / 255
    y_train = np.eye(10)[y_train.astype('int32').flatten()]

    return (x_train, x_test, y_train)

x_train, x_test, y_train = load_mnist()


# In[4]:


print(x_train.shape, x_test.shape, y_train.shape)
print(type(x_train), type(x_test), type(y_train))


# In[5]:


df = pd.DataFrame(y_train)
df.sum()


# In[6]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1, l2, l1_l2

def build_model():

    model = Sequential()
    drop_rate=0.2

    model.add(Dense(1024, input_shape=(784,), activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(drop_rate))
    model.add(Dense(512, activation='relu', kernel_initializer='he_normal',
                    ))
    model.add(Dropout(drop_rate))
    model.add(Dense(512, activation='relu', kernel_initializer='he_normal',))
    model.add(BatchNormalization())
    model.add(Dropout(drop_rate))
    model.add(Dense(1024, activation='relu', kernel_initializer='he_normal',))
    model.add(BatchNormalization())
    model.add(Dropout(drop_rate))
#     model.add(Dense(128, activation='relu', kernel_initializer='he_normal',))
#     model.add(Dropout(drop_rate))    
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[7]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
gen_params = {
    'rotation_range':20,
    'horizontal_flip':True,
    'width_shift_range':0.1,
    'height_shift_range':0.1,
    'channel_shift_range': 0.01,
    'shear_range':5,
    'zoom_range':[0.9, 1.1],
    'samplewise_center': True,
    'samplewise_std_normalization':True
}

def my_generator(x, y, batch_size):
    img_gen = ImageDataGenerator(**gen_params)
    for x_batch, y_batch in img_gen.flow(x.reshape(x.shape[0],28,28,-1), y, batch_size=batch_size):
        inputs = []
#        print(x_batch.shape[0])
        for i in range(x_batch.shape[0]):
            # 次元を１次元に戻して追加
#            print(x_batch[i].shape)
            inputs.append(x_batch[i].reshape(784))
            # plt.imshow(b[i,:,:,0])
            # plt.gray()
            # plt.show()

#        print(np.asarray(inputs).shape)
#         print(y_batch.shape)
        yield np.asarray(inputs), y_batch


# In[8]:


# cv
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau

def run_cv(train, test, target, params={}):
    N = 10
    kf = StratifiedKFold(n_splits=N, random_state=SEED)
    fold_splits = kf.split(train, target.argmax(axis=1))
    tr_scores = []
    val_scores = []
    results = np.zeros((test.shape[0], N))
    i = 1
    
    for tr_idx, val_idx in fold_splits:
        print(f'Start fold {i}/10')
        tr_X, val_X = train[tr_idx, :], train[val_idx, :]
        tr_y, val_y = target[tr_idx, :], target[val_idx, :]
        
        tr_acc, val_acc, test_pred = run_model(tr_X, tr_y, val_X, val_y, test, params)
        tr_scores.append(tr_acc)
        val_scores.append(val_acc)
        results[:, i-1] = test_pred
        i+=1
        
    print('mean acc: ', sum(val_scores)/len(val_scores))
    return results

def run_model(tr_X, tr_y, val_X, val_y, test, params):
    print('Train model')
    batch_size=params['batch_size']
    epochs = params['epochs']

    model = build_model()
    callbacks = [
        EarlyStopping(monitor='val_acc', patience=8),
        ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=5)
    ]
#     model.fit_generator(my_generator(tr_X, tr_y, batch_size),
#                         steps_per_epoch=len(tr_X)//batch_size,
#                         epochs=epochs,
#                         validation_data=my_generator(val_X, val_y, batch_size),
#                         validation_steps=len(val_X)//batch_size,
#                         callbacks=callbacks,
#                         verbose=0)
    model.fit(tr_X, tr_y, batch_size=batch_size, epochs=epochs, validation_data=(val_X, val_y), verbose=2,
              callbacks=callbacks)
    #print(tr_X.reshape(tr_X.shape[0],28,28,-1).shape, val_X.shape)
#     model.fit_generator(train_img_generator.flow(tr_X.reshape(tr_X.shape[0],28,28,-1), tr_y, batch_size=batch_size),
#                         steps_per_epoch=len(tr_X)//batch_size,
#                         epochs=epochs,
#                         validation_data=val_img_generator.flow(val_X.reshape(val_X.shape[0],28,28,-1), val_y, batch_size=batch_size),
#                         validation_steps=len(val_X)//batch_size)
    print('Pred')
    tr_loss, tr_acc = model.evaluate(tr_X, tr_y)
    val_loss, val_acc = model.evaluate(val_X, val_y)
    print(f'[Train] acc:{tr_acc}  loss:{tr_loss}')
    print(f'[Val]   acc:{val_acc} loss:{val_loss}')
    print('Pred 2/2')
    pred_test = np.argmax(model.predict(test), axis=1)
    return tr_acc, val_acc, pred_test
        
params = {'batch_size':32,
          'epochs':100,}
results = run_cv(x_train, x_test, y_train, params)


# In[9]:


#print(results.shape)
submission = pd.DataFrame(results, dtype=int)
submission = submission.apply(lambda x: np.argmax(x.value_counts()), axis=1)
submission.head(20)


# In[11]:


submission.name = 'label'
submission.to_csv('/root/userspace/submission_6.csv', header=True, index_label='id')


# In[ ]:




