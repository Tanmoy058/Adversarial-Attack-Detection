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
#check utils/constants.py for dataset numbers. They are off by one so add 1 to them, cut_data is 0

sess=tf.InteractiveSession()

DATASET_INDEX = 14

X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(DATASET_INDEX,fold_index=None,normalize_timeseries=False) 


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

epoch_num = 50

##################################################################################

model = generate_FCN_model()

ckpt_path = './weights/weights_only.h5'

#model.save_weights(ckpt_path)

#train_model(model, DATASET_INDEX, dataset_prefix='FCN', epochs=epoch_num, batch_size=128, weight_fun=ckpt_path)

#model.save_weights(ckpt_path)

#evaluate_model(model, DATASET_INDEX, dataset_prefix='FCN', batch_size=128)

##################################################################################

model_2 = generate_MLSTM_model()

ckpt_path_2 = './weights/weights_only_2.h5'

#model_2.save_weights(ckpt_path_2)

#train_model(model_2, DATASET_INDEX, dataset_prefix='MLSTM', epochs=epoch_num, batch_size=128, weight_fun=ckpt_path_2)

#model_2.save_weights (ckpt_path_2)

#evaluate_model(model_2, DATASET_INDEX, dataset_prefix='MLSTM', batch_size=128)

##################################################################################

model_3 = generate_MLSTM_attention_model()

ckpt_path_3 = './weights/weights_only_3.h5'

#model_3.save_weights(ckpt_path_3)

#train_model(model_3, DATASET_INDEX, dataset_prefix='MLSTM_attention', epochs=epoch_num, batch_size=128,weight_fun=ckpt_path_3)

#model_3.save_weights(ckpt_path_3)

#evaluate_model(model_3, DATASET_INDEX, dataset_prefix='MLSTM_attention', batch_size=128)

##################################################################################

X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(DATASET_INDEX,fold_index=None,normalize_timeseries=False) 
X_test = pad_sequences(X_test, maxlen=MAX_NB_VARIABLES[DATASET_INDEX], padding='post', truncating='post')
y_test = to_categorical(y_test, len(np.unique(y_test)))

##################################################################################

ckpt_path = ckpt_path_2
model = model_2

##################################################################################


#Next part of notebook
model.load_weights(ckpt_path)
Y_pred = model.predict(X_test)
y_test
Y_pred
t = []
for i in range(len(Y_pred)):
    t.append( sum((max(Y_pred[i])==Y_pred[i]) * y_test[i]) == 0 )
1-sum(t)/len(Y_pred)
'''
    Three Running Mode: whitebox/fake_blackbox/blackbox. 
    Select one of these 3 options as True to proceed.
'''
# whitebox, fake_blackbox, blackbox = True, False, False
# whitebox, fake_blackbox, blackbox = False, True, False
whitebox, fake_blackbox, blackbox = False, False, True
if whitebox:
    use_train_op = True
if fake_blackbox:
    use_train_op, use_grad_op = False, True
if blackbox:
    use_train_op, use_grad_op = False, False

    
'''
    Construct tf-Graph
''' 
CONST_LAMBDA = 10000
SUCCESS_ATTACK_PROB_THRES = 0.00
x = tf.placeholder(tf.float32,[None, MAX_NB_VARIABLE, MAX_TIMESTEPS])
y = tf.placeholder(tf.float32,[None, NB_CLASS])

# In whitebox attack, Var adv is updated thru train_op, while in blackbox attack, adv is updated manually. 
with tf.name_scope('attack'):
    if whitebox:
        adv = tf.Variable(tf.zeros([1, MAX_NB_VARIABLE, MAX_TIMESTEPS]), name = "adv_pert")
    else:
        adv = tf.placeholder(tf.float32, [None, MAX_NB_VARIABLE, MAX_TIMESTEPS])

# specify trainable variable
all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
attack_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='attack')
trainable_vars = tf.trainable_variables()
for var in all_vars:
    if var not in attack_vars:
        trainable_vars.remove(var)
# compute loss
new_x = adv + x
output_x = model(x)
output = model(new_x)

l2dist = tf.reduce_sum(tf.square(adv), [1,2])

real = tf.reduce_sum(y * output, 1)
fake = tf.reduce_max((1 - y) * output, 1)
    
loss1 = CONST_LAMBDA * tf.maximum(-SUCCESS_ATTACK_PROB_THRES, real - fake)
loss2 = l2dist
loss_batch = loss1 + loss2
loss = tf.reduce_sum(loss_batch) # sum over all the batch samples

optimizer = tf.train.AdamOptimizer(0.1)

# replace train_op with manual designed grad_op
if use_train_op:
    train = optimizer.minimize(loss, var_list=trainable_vars)
if use_grad_op:
    grad_op = tf.gradients(loss, adv)

# initialize variables and load target model
sess.run(tf.global_variables_initializer())
model.load_weights(ckpt_path)

print(X_train.shape)

samples = []
signals = []
perturbations = []
perturbation = 0
cur_val = 0
new_val = 0
original_X_train = X_train
sample = model.predict(X_train)[0]
#sample level
for i in range(20):
    #signal level
    for j in range(len(X_train[i])):
        #time step level
        for k in range(len(X_train[i][j])):
            x = False
            while perturbation <= 1000:
                cur_val = np.argmax(sample)
                perturbation += 1
                X_train[i][j][k % len(X_train[i][j])] += 1
                sample = model.predict(X_train)[i]
                new_val = np.argmax(sample)
                if new_val != cur_val:
                    x = True
                    perturbations.append(perturbation)
                    X_train = original_X_train
                    perturbation = 0
                    cur_val = 0
                    new_val = 0
                    break
            if x == False:
                perturbations.append(0)
                perturbation = 0
                X_train = original_X_train
        signals.append(perturbations)
        perturbations = []
    samples.append(signals)
    print(samples)

for i in range(len(samples)):
    for j in range(len(samples[i])):
        print(samples[i][j])
    print("\n\n")


