
# coding: utf-8

# In[1]:


# coding: utf-8

# In[1]:

# for part 4.
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.layers import Dropout, Input, BatchNormalization, Reshape, Lambda
from keras.layers import LSTM, Bidirectional, TimeDistributed, merge
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import numpy as np
from skimage import io as skimio
from skimage import color as skimcolor

import os
import csv
import time
import pickle

# In[2]:


# In[2]:

csv_files_train = "../data/train.csv"
csv_files_eval = "../data/valid.csv"
n_epochs = 5
gpu = "" # help="GPU 0,1 or '' ", default=''

train_batch_size=64
eval_batch_size=64
input_shape=(117, 1669, 1)
# optimizer=SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5) # Darryl's courses way
# optimizer=SGD(lr=1e-3, decay=0.95, momentum=0.9, nesterov=True, clipnorm=5) # from previous code
optimizer=Adam(lr=1e-3, decay=0.95)
alphabet=" !\"#&'()*+,-./0123456789:;<=>?ABCDEFGHIJKLMNOPQRSTUVWXY[]_abcdefghijklmnopqrstuvwxyz|~"
alphabet_codes = list(range(len(alphabet)))
n_classes = len(alphabet)
csv_delimiter='\t'
keep_prob_dropout = 0.7

save_model_file = "model1.h5"
pickle_file = "model1.pkl"

if not os.path.exists('results'):
    os.makedirs('results')


# ## Create data generator

# In[3]:

char_map_str = """
<SPACE> 0
! 1
" 2
# 3
& 4
' 5
( 6
) 7
* 8
+ 9
, 10
- 11
. 12
/ 13
0 14
1 15
2 16
3 17
4 18
5 19
6 20
7 21
8 22
9 23
: 24
; 25
< 26
= 27
> 28
? 29
A 30
B 31
C 32
D 33
E 34
F 35
G 36
H 37
I 38
J 39
K 40
L 41
M 42
N 43
O 44
P 45
Q 46
R 47
S 48
T 49
U 50
V 51
W 52
X 53
Y 54
[ 55
] 56
_ 57
a 58
b 59
c 60
d 61
e 62
f 63
g 64
h 65
i 66
j 67
k 68
l 69
m 70
n 71
o 72
p 73
q 74
r 75
s 76
t 77
u 78
v 79
w 80
x 81
y 82
z 83
| 84
~ 85
"""
# the "blank" character is mapped to 28

char_map = {}
index_map = {}
for line in char_map_str.strip().split('\n'):
    ch, index = line.split()
    char_map[ch] = int(index)
    index_map[int(index)+1] = ch
index_map[2] = ' '


# In[4]:

def text_to_int_sequence(text):
    """ Convert text to an integer sequence """
    int_sequence = []
    for c in text:
        if c == ' ':
            ch = char_map['<SPACE>']
        else:
            ch = char_map[c]
        int_sequence.append(ch)
    return int_sequence

def int_sequence_to_text(int_sequence):
    """ Convert an integer sequence to text """
    text = []
    for c in int_sequence:
        ch = index_map[c]
        text.append(ch)
    return text


# In[5]:

# import training data
with open(csv_files_train) as f:
    readr = csv.reader(f, delimiter="\t")
    train = [row for row in readr]

train = train[:1000]

tr_imgs = np.array([skimio.imread(r[0]) for r in train])
tr_imgs = np.expand_dims(tr_imgs, 3)
tr_len = len(tr_imgs)

max_string_length = max([len(r[1]) for r in train])
tr_labs = np.ones((tr_len, max_string_length)) * 86
tr_label_len = np.zeros((tr_len, 1))
for i in range(tr_len):
    label = np.array(text_to_int_sequence(train[i][1])) 
    tr_labs[i, :len(label)] = label
    tr_label_len[i] = len(label)


# In[7]:



# In[3]:


# # Start making the model

# In[8]:

inputs = Input(shape=input_shape)


# ## The deep cnn part

# In[9]:

# conv1 - maxPool2x2
conv1 = Conv2D(64, (3,3), activation="relu", input_shape=input_shape,
                 padding="same")(inputs)
conv1 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv1)

# conv2 - maxPool2x2
conv2 = Conv2D(128, (3,3), activation="relu", input_shape=input_shape,
                 padding="same")(conv1)
conv2 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv2)

# conv3 - w/batch-norm
conv3 = Conv2D(256, (3,3), input_shape=input_shape, padding="same")(conv2)
conv3 = BatchNormalization(axis=-1)(conv3)
conv3 = Activation("relu")(conv3)

# conv4 - maxPool 2x1
conv4 = Conv2D(256, (3,3), activation="relu", input_shape=input_shape,
                 padding="same")(conv3)
conv4 = MaxPooling2D(pool_size=(2, 1), padding="same")(conv4)

# conv5 - w/batch-norm
conv5 = Conv2D(512, (3,3), input_shape=input_shape, padding="same")(conv4)
conv5 = BatchNormalization(axis=-1)(conv5)
conv5 = Activation("relu")(conv5)

# conv6 - maxPool 2x1
conv6 = Conv2D(512, (3,3), activation="relu", input_shape=input_shape,
                 padding="same")(conv5)
conv6 = MaxPooling2D(pool_size=(2, 1), padding="same")(conv6)

# conv 7 - w/batch-norm
conv7 = Conv2D(512, (2,2), input_shape=input_shape, padding="valid")(conv6)
conv7 = BatchNormalization(axis=-1)(conv7)
conv7 = Activation("relu")(conv7)

# reshape output
# from [batch, height, width, features]
# to [batch, width, height x features]
shp = Model(inputs=[inputs], outputs=[conv7]).output_shape
conv_out = Reshape((shp[2], shp[1]*shp[3]))(conv7)


# ## Bidirectional LSTM

# In[10]:

bi1 = Bidirectional(LSTM(256, return_sequences=True))(conv_out)
bi2 = Bidirectional(LSTM(256, return_sequences=True))(bi1)

bi_out = Dropout(keep_prob_dropout)(bi2)
bi_out = TimeDistributed(Dense(n_classes))(bi_out)

soft = Activation("softmax")(bi_out)


# ## CTC loss

# In[11]:

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# In[12]:

labels = Input(name='the_labels', shape=[max_string_length], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([soft, labels, input_length, label_length])


# In[13]:


# ## Test the model

# In[4]:


model = Model(inputs=[inputs, labels, input_length, label_length],
             outputs=[loss_out, soft])
if os.path.exists('results/'+save_model_file) and os.path.isfile('results/'+save_model_file):
    model.load_weights("results/"+save_model_file)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)


# In[ ]:



shp2 = np.ones(shape=(len(tr_imgs), 1)) * 400 #val_imgs.shape[2]
preds = model.predict(x = [tr_imgs, tr_labs, shp2, tr_label_len], verbose=1)
pickle.dump(preds, open("preds.pkl", "wb"))
# In[ ]:

# shp1 = np.ones(shape=(len(tr_imgs), 1)) * 400 #tr_imgs.shape[2]
# checkpointer = ModelCheckpoint(filepath="results/"+save_model_file, verbose=0, save_weights_only=True)
# hist = model.fit(x = [val_imgs, val_labs, shp2, val_label_len],
#                  y = tr_labs, batch_size = 16, epochs = n_epochs,
#                  verbose = 1, callbacks = [checkpointer])
# #                  validation_data = ([val_imgs, val_labs, shp2, shp2], val_labs))

