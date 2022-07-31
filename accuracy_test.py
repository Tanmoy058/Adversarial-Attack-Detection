from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable
from utils.layer_utils import AttentionLSTM
import tensorflow as tf
import math
import numpy as np
from tqdm import tqdm
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from utils.generic_utils import load_dataset_at
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from utils.constants import MAX_NB_VARIABLES, MAX_TIMESTEPS_LIST

sess=tf.InteractiveSession()

#check utils/constants.py for dataset numbers. They are off by one so add 1 to them, cut_data is 0
DATASET_INDEX = 0

zeros_num = 10000

if DATASET_INDEX == 6:
    zeros_num = 3549
elif DATASET_INDEX == 14:
    zeros_num = 1521
elif DATASET_INDEX == 12:
    zeros_num = 1152

MAX_TIMESTEPS = MAX_TIMESTEPS_LIST[DATASET_INDEX]
MAX_NB_VARIABLE = MAX_NB_VARIABLES[DATASET_INDEX]
NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]

TRAINABLE = True

def generate_MLSTM_model():
    ip = Input(shape=(MAX_NB_VARIABLE, MAX_TIMESTEPS))

    x = Masking()(ip)
    x = LSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()

    # add load model code here to fine-tune

    return model

def generate_MLSTM_attention_model():
    ip = Input(shape=(MAX_NB_VARIABLE, MAX_TIMESTEPS))
    # stride = 10

    # x = Permute((2, 1))(ip)
    # x = Conv1D(MAX_NB_VARIABLES // stride, 8, strides=stride, padding='same', activation='relu', use_bias=False,
    #            kernel_initializer='he_uniform')(x)  # (None, variables / stride, timesteps)
    # x = Permute((2, 1))(x)

    #ip1 = K.reshape(ip,shape=(MAX_TIMESTEPS,MAX_NB_VARIABLES))
    #x = Permute((2, 1))(ip)
    x = Masking()(ip)
    x = AttentionLSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()

    # add load model code here to fine-tune

    return model

def generate_FCN_model():
    ip = Input(shape=(MAX_NB_VARIABLE, MAX_TIMESTEPS))

    #x = Masking()(ip)
    #x = LSTM(8)(x)
    #x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    #x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(y)

    model = Model(ip, out)
    model.summary()

    # add load model code here to fine-tune

    return model
    
def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = input._keras_shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se

epoch_num = 200

##################################################################################

model = generate_FCN_model()

ckpt_path = './weights/weights_only.h5'

#model.save_weights(ckpt_path)

#train_model(model, DATASET_INDEX, dataset_prefix='FCN', epochs=epoch_num, batch_size=128, weight_fun=ckpt_path)

#model.save_weights(ckpt_path)

#evaluate_model(model, DATASET_INDEX, dataset_prefix='FCN', batch_size=128)

##################################################################################

model_2 = generate_MLSTM_attention_model()

ckpt_path_2 = './weights/weights_only_2.h5'

#model_2.save_weights(ckpt_path_2)

#train_model(model_2, DATASET_INDEX, dataset_prefix='MLSTM', epochs=epoch_num, batch_size=128, weight_fun=ckpt_path_2)

#model_2.save_weights (ckpt_path_2)

#evaluate_model(model_2, DATASET_INDEX, dataset_prefix='MLSTM', batch_size=128)

##################################################################################


X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(DATASET_INDEX,fold_index=None,normalize_timeseries=False) 
X_test = pad_sequences(X_test, maxlen=MAX_NB_VARIABLES[DATASET_INDEX], padding='post', truncating='post')
y_test = to_categorical(y_test, len(np.unique(y_test)))


print("\n\n")
print(y_train)
print("\n\n")
##################################################################################

ckpt_path = ckpt_path
model = model

##################################################################################

acc_list = []
vals_list = []
pred_list = []
pred_2_list = []
match_list = []


model.load_weights(ckpt_path)
model_2.load_weights(ckpt_path_2)

Y_pred = model.predict(X_test)
Y_pred_2 = model_2.predict(X_test)



for i in range(len(Y_pred)):
    vals_list.append(np.argmax(y_test[i]))
    pred_list.append(np.argmax(Y_pred[i]))
    pred_2_list.append(np.argmax(Y_pred_2[i]))

for i in range(len(vals_list)):
    if(vals_list[i] != pred_list[i]):
        acc_list.append(vals_list[i])
    if(pred_list[i] == pred_2_list[i]):
        match_list.append([i, pred_list[i]])

print("Number of instances of class 1:")
print(vals_list.count(0))
print("\nAccuracy for class 1: ")
print((vals_list.count(0) - acc_list.count(0))/(vals_list.count(0)))
print("Number of instances of class 2:")
print(vals_list.count(1))
print("\nAccuracy for class 2: ")
print((vals_list.count(1) - acc_list.count(1))/(vals_list.count(1)))
print("Number of instances of class 3:")
print(vals_list.count(2))
print("\nAccuracy for class 3: ")
print((vals_list.count(2) - acc_list.count(2))/(vals_list.count(2)))
print("Number of instances of class 4:")
print(vals_list.count(3))
print("\nAccuracy for class 4: ")
print((vals_list.count(3) - acc_list.count(3))/(vals_list.count(3)))
print("Number of instances of class 5:")
print(vals_list.count(4))
print("\nAccuracy for class 5: ")
print((vals_list.count(4) - acc_list.count(4))/(vals_list.count(4)))
print("Number of instances of class 6:")
print(vals_list.count(6))
print("\nAccuracy for class 6: ")
print((vals_list.count(5) - acc_list.count(5))/(vals_list.count(5)))
print("Number of instances of class 7:")
print(vals_list.count(6))
print("\nAccuracy for class 7: ")
print((vals_list.count(6) - acc_list.count(6))/(vals_list.count(6)))

print("Total number of matching indices is: " + str(len(match_list)) + "\n")

for i in range(len(match_list)):
    print("Match found at index: " + str(match_list[i][0]) + " for class: " + str(match_list[i][1]))
