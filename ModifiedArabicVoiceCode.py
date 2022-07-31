from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from tensorflow.keras.utils import *
from tensorflow.keras.models import load_model
from tensorflow import keras

from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable
from utils.layer_utils import AttentionLSTM
import os
import tensorflow as tf
import cProfile
import math
import numpy as np
from tqdm import tqdm

tf.compat.v1.disable_v2_behavior()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

DATASET_INDEX = 6

MAX_TIMESTEPS = MAX_TIMESTEPS_LIST[DATASET_INDEX]
MAX_NB_VARIABLES = MAX_NB_VARIABLES[DATASET_INDEX]
NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]

TRAINABLE = True

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



import json


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

sess=tf.compat.v1.InteractiveSession()

tlab = tf.constant(np.ones(7), dtype=tf.float32)
opt = tf.placeholder(tf.float32,[3, 7])
real = tf.reduce_sum((tlab)*opt,1)
ll = tf.maximum(0.0, real)

sess.run(ll, feed_dict={opt:np.ones((3,7))})

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from utils.generic_utils import load_dataset_at
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from utils.constants import MAX_NB_VARIABLES, MAX_TIMESTEPS_LIST
from tensorflow.keras.models import load_model, model_from_json
from sklearn.metrics import confusion_matrix

#model = model_from_json(model_json)
#model.load_weights('./model_weights_2.h5')
ckpt_path = "./weights/arabic_weights.h5"
model = load_model(ckpt_path)

X_test = np.load("./data/arabic_voice/X_test.npy")
y_test = np.load("./data/arabic_voice/y_test.npy")
X_train = np.load("./data/arabic_voice/X_train.npy")
y_train = np.load("./data/arabic_voice/y_train.npy")
nb_classes = len(np.unique(y_test))
batch_size = 16

Y_train = keras.utils.to_categorical(y_train-1, nb_classes)
Y_test = keras.utils.to_categorical(y_test-1, nb_classes)

X_train_mean = X_train.mean()
X_train_std = X_train.std()
X_train = (X_train - X_train_mean)/(X_train_std)

X_test = (X_test - X_train_mean)/(X_train_std)
X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))  
model.load_weights(ckpt_path)

Y_pred = model.predict(X_test)
y_test
Y_pred
t = []
for i in range(len(Y_pred)):
    t.append( sum((max(Y_pred[i])==Y_pred[i]) * y_test[i]) == 0 )
print(1-sum(t)/len(Y_pred))


tlab = tf.constant(np.zeros(7), dtype=tf.float32)
opt = tf.constant(np.zeros((3,7)), dtype=tf.float32)
real = tf.reduce_sum((tlab)*opt,1)


import tensorflow as tf
import math
import numpy as np
from tqdm import tqdm

sess=tf.compat.v1.InteractiveSession()

whitebox, fake_blackbox, blackbox = False, True, False
if whitebox:
    use_train_op = True
if fake_blackbox:
    use_train_op, use_grad_op = False, True
if blackbox:
    use_train_op, use_grad_op = False, False

    

CONST_LAMBDA = 10000
SUCCESS_ATTACK_PROB_THRES = 0.00
x = tf.compat.v1.placeholder(tf.float32,[1,39,91,1])
y = tf.compat.v1.placeholder(tf.float32,[None, nb_classes])


with tf.name_scope('attack'):
    if whitebox:
        adv = tf.compat.v1.Variable(tf.zeros([1,39,91,1]), name = "adv_pert")
    else:
        adv = tf.compat.v1.placeholder(tf.float32, [1,39,91,1])


all_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
attack_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='attack')
trainable_vars = tf.compat.v1.trainable_variables()
for var in all_vars:
    if var not in attack_vars:
        trainable_vars.remove(var)


new_x = adv + x
output_x = model(x)
output = model(new_x)

l2dist = tf.reduce_sum(tf.square(adv), [1,2])

real = tf.reduce_sum(y * output, 1)
fake = tf.reduce_max((1 - y) * output, 1)
    
loss1 = CONST_LAMBDA * tf.maximum(-SUCCESS_ATTACK_PROB_THRES, real - fake)
loss2 = l2dist
loss_batch = loss1 + loss2
loss = tf.reduce_sum(loss_batch)

optimizer = tf.compat.v1.train.AdamOptimizer(0.1)


if use_train_op:
    train = optimizer.minimize(loss, var_list=trainable_vars)
if use_grad_op:
    grad_op = tf.gradients(loss, adv)


sess.run(tf.compat.v1.global_variables_initializer())
model.load_weights(ckpt_path)

def attack_summary(BB):
    SUCCESS_ATTACK_PROB_THRES = BB.SUCCESS_ATTACK_PROB_THRES
    CONST_LAMBDA = BB.CONST_LAMBDA
    learning_rate = BB.learning_rate
    h = BB.h
    MAX_ITE = BB.MAX_ITE
    beta1 = BB.beta1
    beta2 = BB.beta2
    epsilon = BB.epsilon
    success_count = BB.success_count
    fail_count = BB.fail_count
    invalid_count = BB.invalid_count
    query_summary = BB.query_summary
    adv_summary = BB.adv_summary

    algo_params = {"SUCCESS_ATTACK_PROB_THRES": SUCCESS_ATTACK_PROB_THRES, "CONST_LAMBDA":CONST_LAMBDA, \
                  "learning_rate":learning_rate, "h":h, "MAX_ITE": MAX_ITE, "beta1":beta1, "beta2":beta2, "epsilon":epsilon}

    (success_count, fail_count, invalid_count)
    success_count/(fail_count+success_count)
    fail_count/(fail_count+success_count)
    invalid_count/X_test.shape[0]
    sum(list(query_summary.values()))/success_count
    avg_l2_percent = []
    avg_l2_abs = []
    for success_index in query_summary.keys():
        if query_summary[success_index] != 500 and sum(sum(X_test[success_index]**2)) > 10:
            avg_l2_percent.append( math.sqrt( sum(sum(adv_summary[success_index][0]**2)) / sum(sum(X_test[success_index]**2)) ) )
            avg_l2_abs.append( math.sqrt( sum(sum(adv_summary[success_index][0]**2)) ) )

    avg_l2 = sum(avg_l2_percent) / len(avg_l2_percent)
    avg_l2_ = sum(avg_l2_abs) / len(avg_l2_abs)
    
    a = []
    for i in BB.query_summary.keys():
        if query_summary[i] != 500:
            a.append(sum(sum(abs(BB.adv_summary[i][0][:,:]) > 0)) /2/ max(sum(abs(X_test[i][0])>0), sum(abs(X_test[i][1])>0), 1))

    avg_l0 = sum(a) / len(a)
    
    res_summary = {"success_count": success_count, \
            "fail_count": fail_count, \
            "invalid_count": invalid_count, \
            "target_model_accuracy": 1-invalid_count/X_test.shape[0], \
            "success_rate": success_count/(fail_count+success_count), \
            "avg_ite": sum( [x for x in list(query_summary.values()) if x!=500])/success_count, \
            "avg_l2": avg_l2, \
            "avg_l2_": avg_l2_, \
            "avg_l0": avg_l0
           }

    print("\n-------Algorithm parameters-------\n")
    print("SUCCESS_ATTACK_PROB_THRES: {SUCCESS_ATTACK_PROB_THRES}, \n\
    CONST_LAMBDA: {CONST_LAMBDA}, \n\
    LEARNING_RATE: {learning_rate}, \n\
    discretization step h: {h}, \n\
    beta_1: {beta1}, \n\
    beta_2: {beta2}, \n\
    epsilon: {epsilon}".format(**algo_params))
    print("\n-------operation result-------\n")
    print("Successful times: {success_count}, \n\
    number of failures: {fail_count}, \n\
    Target model accuracy: {target_model_accuracy}, \n\
    attack Success rate: {success_rate}, \n\
    Average disturbancel2: {avg_l2_}, \n\
    Average disturbance l2 ratio: {avg_l2}, \n\
    Average disturbance l0 ratio: {avg_l0}, \n\
    Average number of iterations: {avg_ite}".format(**res_summary))
    
    
### BATCH ATTACK CODE

success_count = 0
query_summary = {}
adv_summary = {}
fail_count = 0
invalid_count = 0

for t in range(X_test.shape[0]):
# for t in range(20):
    
    print("\n start attacking target", t, "...")
    
    mt = 0               # accumulator m_t in Adam
    vt = 0               # accumulator v_t in Adam
    beta1=0.9            # parameter beta_1 in Adam
    beta2=0.999          # parameter beta_2 in Adam
    learning_rate = 1e-1 # learning rate
    epsilon = 1e-8       # parameter epsilon in Adam
    h = 0.0001           # discretization constant when estimating numerical gradient
    batch_size = 1       # batch size
    MAX_ITE = 200        # maximum number of iterations


    real_adv = np.zeros([1,39,91,1])   # initial adversarial perturbation, the trainable variable
    X = X_test[t:t+1]           # target sample X
    Y = Y_test[t:t+1]           # target sample's lable Y
    
    non_zeros = max(sum(abs(X[0][0][0])>0), sum(abs(X[0][0][1])>0), 1)
    

    pred_y = model.predict(X)
    if sum((max(pred_y[0])==pred_y[0]) * Y[0]) == 0:
        print("not a valid target.")
        invalid_count += 1
        continue
        


    if blackbox:
        X3 = np.repeat(X, 3, axis=0)
        Y3 = np.repeat(Y, 3, axis=0)



    for epoch in range(1, MAX_ITE+1):

        if use_train_op: 

            sess.run(train, feed_dict={x:X,y:Y})
            adv1, output1, l2dist1, real1, fake1, loss1, new_x1 = sess.run([adv, output, l2dist, real, fake, loss, new_x], feed_dict={x:X,y:Y})
            print(l2dist1, adv1, X, new_x1)
        else: 

            if use_grad_op: 

                true_grads, los, l2s, los1, los2, scores, scores_x, nx, adv1 = sess.run([grad_op, loss, l2dist, loss1, loss2, output, output_x, new_x, adv], feed_dict={adv: real_adv, x:X, y:Y})

                true_grads[0][0][0:2,non_zeros:]=0

                grad = true_grads[0].reshape(-1)

            else: 

                var = np.repeat(real_adv, batch_size*3, axis=0)
                var_size = real_adv[0].size # (2, 506)

                update_indice = np.random.choice(list(range(non_zeros))+list(range(506,506+non_zeros)), 1, replace=False) 

                for i in range(batch_size):
                    var[batch_size * 1 + i].reshape(-1)[update_indice[0]] += h
                    var[batch_size * 2 + i].reshape(-1)[update_indice[0]] -= h

                los, l2s, los_b, scores, nx, adv1 = sess.run([loss, l2dist, loss_batch, output, new_x, adv], feed_dict={adv: var, x:X3, y:Y3})

                grad = np.zeros(real_adv.reshape(-1).shape)

                for i in range(batch_size):
                    grad[update_indice[0]] += los_b[batch_size * 1 + i]- los_b[batch_size * 2 + i]
                grad[update_indice[0]] /= 2*h

            mt = beta1 * mt + (1 - beta1) * grad
            vt = beta2 * vt + (1 - beta2) * np.square(grad)
            corr = (math.sqrt(1 - beta2 ** epoch)) / (1 - beta1 ** epoch)

            m = real_adv.reshape(-1)
            m -= learning_rate * corr * (mt / (np.sqrt(vt) + epsilon))
            real_adv = m.reshape(real_adv.shape)

            if use_grad_op: 
                if epoch == MAX_ITE:
                    print("attack failed!")
                    fail_count += 1
                    break
                if sum((scores[0] == max(scores[0]))*Y[0])==0:
                    print("attack successed! with ite =", epoch)
                    success_count += 1
                    query_summary[t] = epoch
                    adv_summary[t] = adv1
                    break

            else:
                if epoch == MAX_ITE:
                    print("attack failed!")
                    fail_count += 1
                    break
                if sum((scores[0] == max(scores[0]))*Y[0])==0:
                    print("attack successed! with ite =", epoch)
                    success_count += 1
                    query_summary[t] = epoch
                    adv_summary[t] = real_adv
                    break

                    

algo_params = {"SUCCESS_ATTACK_PROB_THRES": SUCCESS_ATTACK_PROB_THRES, "CONST_LAMBDA":CONST_LAMBDA, \
                  "learning_rate":learning_rate, "h":h, "MAX_ITE": MAX_ITE, "beta1":beta1, "beta2":beta2, "epsilon":epsilon}

(success_count, fail_count, invalid_count)
success_count/(fail_count+success_count)
fail_count/(fail_count+success_count)
invalid_count/X_test.shape[0]
sum(list(query_summary.values()))/success_count
avg_l2_percent = []
avg_l2_abs = []
for success_index in query_summary.keys():
    if query_summary[success_index] != MAX_ITE and sum(sum(X_test[success_index]**2)) > 10:
        avg_l2_percent.append( math.sqrt( sum(sum(adv_summary[success_index][0]**2)) / sum(sum(X_test[success_index]**2)) ) )
        avg_l2_abs.append( math.sqrt( sum(sum(adv_summary[success_index][0]**2)) ) )

avg_l2 = sum(avg_l2_percent) / len(avg_l2_percent)
avg_l2_ = sum(avg_l2_abs) / len(avg_l2_abs)

a = []
for i in query_summary.keys():
    if query_summary[i] != MAX_ITE:
        a.append(sum(sum(abs(adv_summary[i][0][:,:]) > 0)) /2/ max(sum(abs(X_test[i][0])>0), sum(abs(X_test[i][1])>0), 1))
avg_l0 = sum(a) / len(a)

res_summary = {"success_count": success_count, \
               "fail_count": fail_count, \
               "invalid_count": invalid_count, \
               "target_model_accuracy": 1-invalid_count/X_test.shape[0], \
               "success_rate": success_count/(fail_count+success_count), \
               "avg_ite": sum( [x for x in list(query_summary.values()) if x!=500])/success_count, \
               "avg_l2": avg_l2, \
               "avg_l2_": avg_l2_, \
               "avg_l0": avg_l0
                              }

print(res_summary)

#### END OF BATCH ATTACK CODE

### BEGINNING OF BLACKBOX L2 CODE

class BlackboxL2:
    def __init__(self, mode='blackbox', sess=sess, model=model, ckpt_path=ckpt_path, \
                CONST_LAMBDA=10000, prob_thres=0.0, batch_size=1, learning_rate=2e-1, h=1e-4, MAX_ITE=2000, \
                adapt_h=False, adapt_lambda=False, norm_gradient=False):
            self.success_count = 0
    self.query_summary = {}
    self.adv_summary = {}
    self.loss_lambda = {}
    self.fail_count = 0
    self.invalid_count = 0
    
    self.mode = mode
    if self.mode == 'whitebox':
        self.use_train_op = True
    if self.mode == 'fake_blackbox':
        self.use_train_op, self.use_grad_op = False, True
    if self.mode == 'blackbox':
        self.use_train_op, self.use_grad_op = False, False

    self.CONST_LAMBDA = CONST_LAMBDA
    self.SUCCESS_ATTACK_PROB_THRES = prob_thres
    
    if self.mode == 'blackbox':
        self.x = tf.compat.v1.placeholder(tf.float32,[3,39,91,1])
    else:
        self.x = tf.compat.v1.placeholder(tf.float32,[1,39,91,1])
    self.y = tf.compat.v1.placeholder(tf.float32,[None, nb_classes])
    self.const = tf.compat.v1.placeholder(tf.float32)
    self.model = model
    self.sess = sess
    
    self.beta1=0.9                       # parameter beta_1 in Adam
    self.beta2=0.999                     # parameter beta_2 in Adam
    self.learning_rate = learning_rate   # learning rate
    self.epsilon = 1e-8                  # parameter epsilon in Adam
    self.h = h                           # discretization constant when estimating numerical gradient
    self.batch_size = batch_size         # batch size
    self.MAX_ITE = MAX_ITE               # maximum number of iterations
    self.adapt_h = adapt_h
    self.adapt_lambda = adapt_lambda
    self.norm_gradient = norm_gradient

    with tf.name_scope('attack'):
        if self.mode == 'whitebox':
            self.adv = tf.Variable(tf.zeros([1,39,91,1]), name = "adv_pert")
        else:
            self.adv = tf.compat.v1.placeholder(tf.float32, [3,39,91,1])

    all_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    attack_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='attack')
    trainable_vars = tf.compat.v1.trainable_variables()
    for var in all_vars:
        if var not in attack_vars:
            trainable_vars.remove(var)

    self.new_x = self.adv + self.x
    self.output_x = self.model(self.x)
    self.output = self.model(self.new_x)

    self.l2dist = tf.reduce_sum(tf.square(self.adv), [1,2])

    self.real = tf.reduce_sum(self.y * self.output, 1)
    self.fake = tf.reduce_max((1 - self.y) * self.output, 1)

    self.loss1 = self.const * tf.maximum(-self.SUCCESS_ATTACK_PROB_THRES, self.real - self.fake)
    self.loss2 = self.l2dist
    self.loss_batch = self.loss1 + self.loss2
    self.loss = tf.reduce_sum(self.loss_batch) 

    self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)

    if self.use_train_op:
        self.train = optimizer.minimize(self.loss, var_list=trainable_vars)
    if self.use_grad_op:
        self.grad_op = tf.gradients(self.loss, self.adv)

    self.sess.run(tf.compat.v1.global_variables_initializer())
    self.model.load_weights(ckpt_path)

    def reset_summary(self):

        self.success_count = 0
        self.query_summary = {}
        self.adv_summary = {}
        self.loss_lambda = {}
        self.fail_count = 0
        self.invalid_count = 0

    def attack(self, X, Y, t, weight=np.ones((39, 91,1)), option='vanilla'):

        X = X[t:t+1]
        Y = Y[t:t+1]
        weight = np.ones((39,91,1))
        if not self.adapt_lambda:
            return self.attack_lambda(X, Y, self.CONST_LAMBDA, t, weight, option=option)

        flag = 0
        L = self.CONST_LAMBDA
        while flag==0 and L <= 1e6:
            print("attempt to attack with lambda =", L, ":")
            flag,r1,r2,r3,r4,_ = self.attack_lambda(X, Y, L, t, weight, option=option)
            L *= 10
        self.fail_count += 1-flag
        return flag,r1,r2,r3,r4,L/10

    def attack_lambda(self, X, Y, LAMB, t, weight, option):

        mt = np.zeros((3549,))                          # accumulator m_t in Adam
        vt = np.zeros((3549,))                          # accumulator v_t in Adam

        real_adv = np.zeros((1,39,91,1))  


        self.non_zeros = max(sum(abs(X[0][0][0])>0), sum(abs(X[0][0][1])>0), 1)


        p = np.hstack([weight[0,:self.non_zeros][::-1], weight[1,:self.non_zeros][::-1]])
        p /= sum(p)

        pred_y = self.model.predict(X)
        if sum((max(pred_y[0])==pred_y[0]) * Y[0]) == 0:
            print("not a valid target.")
            self.invalid_count += 1
            return -1, 0, real_adv, 0, 0, 0

        if self.mode == 'blackbox':
            X3 = np.repeat(X, 3, axis=0)
            Y3 = np.repeat(Y, 3, axis=0)

        # main loop for the optimization
        num_query = 0
        for epoch in range(1, self.MAX_ITE+1):

            if self.use_train_op: 

                self.sess.run(self.train, feed_dict={x:X,y:Y})
                adv1, output1, l2dist1, real1, fake1, loss1, new_x1 = self.sess.run([self.adv, self.output, self.l2dist, self.real, self.fake, self.loss, self.new_x], feed_dict={self.x:X,self.y:Y,self.const:self.CONST_LAMBDA})
                print(l2dist1, adv1, X, new_x1)

            else:

                if self.use_grad_op: 

                    true_grads, los, l2s, los1, los2, scores, scores_x, nx, adv1 = self.sess.run([self.grad_op, self.loss, self.l2dist, self.loss1, self.loss2, self.output, self.output_x, self.new_x, self.adv], feed_dict={self.adv: real_adv, self.x:X, self.y:Y,self.const:self.CONST_LAMBDA})

                    true_grads[0][0][0:2,self.non_zeros:]=0

                    grad = true_grads[0].reshape(-1)

                else: 

                    var = np.repeat(real_adv, self.batch_size*3, axis=0)
                    var_size = real_adv[0].size

                    a = list(range(self.non_zeros))
                    ep = (1/(1+np.exp(np.sqrt(vt[a]) * -0.1)))**4
                    p1 = ep / sum(ep)

                    if option=='vanilla':
                        update_indice = np.random.choice(a=a, size=1, replace=False) 
                    if option=='mwu-inner':
                        update_indice = np.random.choice(a=a, size=1, replace=False, p=p1) 
                    if option=='mwu-outer':
                        update_indice = np.random.choice(a=a, size=1, replace=False, p=p) 
                    if option=='mwu-mix':
                        p2 = p*p1
                        update_indice = np.random.choice(a=a, size=1, replace=False, p=p2/sum(p2))

                    grad = np.zeros(real_adv.reshape(-1).shape)
                    hh = self.h
                    while sum(grad)==0 and hh < 10: 
                        for i in range(self.batch_size):
                            var[self.batch_size * 1 + i].reshape(-1)[update_indice[0]] += hh
                            var[self.batch_size * 2 + i].reshape(-1)[update_indice[0]] -= hh

                        los, l2s, los_b, scores, nx, adv1 = self.sess.run([self.loss, self.l2dist, self.loss_batch, self.output, self.new_x, self.adv], feed_dict={self.adv: var, self.x:X3, self.y:Y3,self.const:LAMB})
                        num_query += 3

                        grad = np.zeros(real_adv.reshape(-1).shape)

                        for i in range(self.batch_size):
                            grad[update_indice[0]] += los_b[i][self.batch_size * 1 + i]- los_b[i][self.batch_size * 2 + i]
                        grad[update_indice[0]] /= hh
                        if not self.adapt_h:
                            break
                        hh *= 2
                    if self.norm_gradient:
                        norm_grad = math.sqrt(sum(grad**2))
                        if norm_grad > 0:
                            grad /= norm_grad

                mt = self.beta1 * mt + (1 - self.beta1) * grad
                vt = self.beta2 * vt + (1 - self.beta2) * np.square(grad)

                corr = (math.sqrt(1 - self.beta2 ** epoch)) / (1 - self.beta1 ** epoch)

                m = real_adv.reshape(-1)
                m -= self.learning_rate * corr * (mt / (np.sqrt(vt) + self.epsilon))
                real_adv = m.reshape(real_adv.shape)

                if self.use_grad_op: 
                    if epoch == self.MAX_ITE:
                        print("attack failed!")
                        self.fail_count += 1
                        return 0
                    if sum((scores[0] == max(scores[0]))*Y[0])==0:
                        print("attack succeeded! with ite =", epoch)
                        self.success_count += 1
                        self.query_summary[t] = epoch
                        self.adv_summary[t] = adv1
                        return 1


                else:
                    if epoch == self.MAX_ITE:
                        print("attack failed!")
                        if not self.adapt_lambda:
                            self.fail_count += 1
                        self.query_summary[t] = epoch
                        self.adv_summary[t] = real_adv
                        self.loss_lambda[t] = LAMB
                        return 0, scores, real_adv, math.sqrt(sum(sum(real_adv[0]**2))), epoch, LAMB
                    if sum((scores[0] == max(scores[0]))*Y[0])==0:
                        print("attack succeeded! with ite =", epoch)
                        self.success_count += 1
                        self.query_summary[t] = epoch
                        self.adv_summary[t] = real_adv
                        self.loss_lambda[t] = LAMB
                        return 1, scores, real_adv, math.sqrt(sum(sum(real_adv[0]**2))), epoch, LAMB

                    
BB11 = BlackboxL2(mode='blackbox', sess=sess, model=model, ckpt_path=ckpt_path, \
CONST_LAMBDA=10000, prob_thres=0.0, batch_size=1, learning_rate=1, h=1e-4, MAX_ITE=350, \
adapt_h=True, adapt_lambda=False)
weight = np.ones((39,91,1))


for t in range(len(y_test)):
    print("\n start attacking target", t, "...")
    success_flag,pred_score,pert,pert_norm,_,_=BB11.attack(X_test, y_test, t, weight, option='vanilla')

    
    l = max(sum(abs(X_test[t][0])>0), sum(abs(X_test[t][1])>0), 1)

    if success_flag1 == 1:
        p = abs(pert1.copy())
        p[0][0,:l] = p[0][0,:l][::-1] # reverse
        p[0][1,:l] = p[0][1,:l][::-1] # reverse
        weight1 += abs(p[0]) / sum(sum(abs(p[0])))

### END OF BLACKBOXL2

### START OF BLACKBOXL0
                    
class BlackboxL0:
    def __init__(self, mode='blackbox', sess=sess, model=model, ckpt_path=ckpt_path, \
                 CONST_LAMBDA=10000, prob_thres=0.0, batch_size=1, learning_rate=2e-1, h=1e-4, MAX_ITE=2000, \
                 adapt_h=False, adapt_lambda=False, norm_gradient=False):

        self.success_count = 0
        self.query_summary = {}
        self.adv_summary = {}
        self.loss_lambda = {}
        self.fail_count = 0
        self.invalid_count = 0
        
        self.mode = mode
        if self.mode == 'whitebox':
            self.use_train_op = True
        if self.mode == 'fake_blackbox':
            self.use_train_op, self.use_grad_op = False, True
        if self.mode == 'blackbox':
            self.use_train_op, self.use_grad_op = False, False


        self.CONST_LAMBDA = CONST_LAMBDA
        self.SUCCESS_ATTACK_PROB_THRES = prob_thres
        
        if self.mode == 'blackbox':
            self.x = tf.compat.v1.placeholder(tf.float32,[3,39,91,1])
        else:
            self.x = tf.compat.v1.placeholder(tf.float32,[1,39,91,1])
        self.y = tf.compat.v1.placeholder(tf.float32,[None, nb_classes])
        self.const = tf.compat.v1.placeholder(tf.float32)
        self.model = model
        self.sess = sess
        
        self.beta1=0.9                       # parameter beta_1 in Adam
        self.beta2=0.999                     # parameter beta_2 in Adam
        self.learning_rate = learning_rate   # learning rate
        self.epsilon = 1e-8                  # parameter epsilon in Adam
        self.h = h                           # discretization constant when estimating numerical gradient
        self.batch_size = batch_size         # batch size
        self.MAX_ITE = MAX_ITE               # maximum number of iterations
        self.adapt_h = adapt_h
        self.adapt_lambda = adapt_lambda
        self.norm_gradient = norm_gradient

        with tf.name_scope('attack'):
            if self.mode == 'whitebox':
                self.adv = tf.Variable(tf.zeros([1,39,91,1]), name = "adv_pert")
            else:
                self.adv = tf.compat.v1.placeholder(tf.float32, [3, 39, 91,1])

        all_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        attack_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='attack')
        trainable_vars = tf.compat.v1.trainable_variables()
        for var in all_vars:
            if var not in attack_vars:
                trainable_vars.remove(var)

        self.new_x = self.adv + self.x
        self.output_x = self.model(self.x)
        self.output = self.model(self.new_x)

        self.l2dist = tf.reduce_sum(tf.square(self.adv), [1,2])

        self.real = tf.reduce_sum(self.y * self.output, 1)
        self.fake = tf.reduce_max((1 - self.y) * self.output, 1)

        self.loss1 = self.const * tf.maximum(-self.SUCCESS_ATTACK_PROB_THRES, self.real - self.fake)
        self.loss2 = self.l2dist
        self.loss_batch = self.loss1 + self.loss2
        self.loss = tf.reduce_sum(self.loss_batch) 

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)

        if self.use_train_op:
            self.train = optimizer.minimize(self.loss, var_list=trainable_vars)
        if self.use_grad_op:
            self.grad_op = tf.gradients(self.loss, self.adv)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.model.load_weights(ckpt_path)
    
    def reset_summary(self):
        
        self.success_count = 0
        self.query_summary = {}
        self.adv_summary = {}
        self.loss_lambda = {}
        self.fail_count = 0
        self.invalid_count = 0
    
    def yield_shrink_indice(self, pert, method='min'):
        
        if method=='min':
            pert[0][pert[0]==0] = 1e8
            shrink_indice = np.argmin(abs(pert[0]))
            pert[0][pert[0]==1e8] = 0
        elif method=='max':
            shrink_indice = np.argmax(abs(pert[0]))
        elif method=='vt-min':
            self.vt[self.vt==0] = 1e8
            shrink_indice = np.argmin(self.vt)
            self.vt[self.vt==1e8] = 0
            self.vt[shrink_indice] = 0
        elif method=='vt-max':
            shrink_indice = np.argmax(self.vt)
            self.vt[shrink_indice] = 0


        return shrink_indice
    
    def attack_L0(self, X, Y, t, weight=np.ones((39, 91, 1))):
        
        flag, scores, pert, l2dist, epoch, valid_lbd = self.attack(X, Y, t, np.zeros([1, 39, 91, 1]), weight=weight)
        
        self.vt = scores 
        
        if flag < 1:
            return flag, scores, pert, l2dist, epoch, valid_lbd
        
        l2_anchor = l2dist
        support_mask = [0, 1]
        while l2dist < l2_anchor * 1.1 and len(support_mask) > 1:
            support_mask = [i for i in list(range(self.non_zeros))+list(range(506,506+self.non_zeros)) if abs(pert.reshape(-1)[i]) > 0]
            
            shrink_indice = self.yield_shrink_indice(pert, method='vt-max')
            print(shrink_indice)
            
            if shrink_indice in support_mask:
                support_mask.remove(shrink_indice)
                pert_shrink = pert.copy()
                pert_shrink.reshape(-1)[shrink_indice] = 0
            else:
                print("unexpected shrink indice detected.")
                return 0,0,0,0,0,0
            
            print("L2 -> L0 squeeze: L2=", l2dist, ", L0=", len(support_mask), ", epoch=", epoch)

            flag_1, scores_1, pert_1, l2dist_1, epoch_1, valid_lbd_1 = self.attack_lambda(X, Y, t, valid_lbd, pert_shrink, support_mask, record_result=False, weight=weight)
            
            if flag_1 < 1:
                self.query_summary[t] = epoch
                self.adv_summary[t] = pert
                self.loss_lambda[t] = valid_lbd
                return flag, scores, pert, l2dist, epoch, valid_lbd
            else:
                flag, scores, pert, l2dist, epoch, valid_lbd = flag_1, scores_1, pert_1, l2dist_1, epoch+epoch_1, valid_lbd_1
        
        self.query_summary[t] = epoch
        self.adv_summary[t] = pert
        self.loss_lambda[t] = valid_lbd

        return flag, scores, pert, l2dist, epoch, valid_lbd
        
        

    def attack(self, X, Y, t, init_pert=np.zeros([1, 39,91,1]), weight=np.ones((39,91,1))):
        
        
        if not self.adapt_lambda:
            return self.attack_lambda(X, Y, t, self.CONST_LAMBDA, init_pert, weight=weight)

        flag = 0
        L = self.CONST_LAMBDA
        while flag==0 and L <= 1e6:
            print("attempt to attack with lambda =", L, ":")
            flag,r1,r2,r3,r4,_ = self.attack_lambda(X, Y, t, L, init_pert, weight=weight)
            L *= 10
        self.fail_count += 1-flag
        return flag,r1,r2,r3,r4,L/10
        
    def attack_lambda(self, X, Y, t, LAMB, init_pert=np.zeros([1, 39, 91,1]), support_mask=None, record_result=True, weight=np.ones((39,91,1))):
        
        X = X[t:t+1]
        Y = Y[t:t+1]
        
        mt = 0                          
        vt = 0                         

        real_adv = init_pert  
        
        self.non_zeros = max(sum(abs(X[0][0][0])>0), sum(abs(X[0][1][0])>0), 1)
        
        if support_mask is None:
            p = np.hstack([weight[0,:self.non_zeros][::-1], weight[1,:self.non_zeros][::-1]])
            p = p[0]
            p /= sum(p)
            
        pred_y = self.model.predict(X)
        if sum((max(pred_y[0])==pred_y[0]) * Y[0]) == 0:
            print("not a valid target.")
            self.invalid_count += 1
            return -1, 0, real_adv, 0, 0, 0
        
        if self.mode == 'blackbox':
            X3 = np.repeat(X, 3, axis=0)
            Y3 = np.repeat(Y, 3, axis=0)

        for epoch in range(1, self.MAX_ITE+1):

            if self.use_train_op: 

                self.sess.run(self.train, feed_dict={x:X,y:Y})
                adv1, output1, l2dist1, real1, fake1, loss1, new_x1 = self.sess.run([self.adv, self.output, self.l2dist, self.real, self.fake, self.loss, self.new_x], feed_dict={self.x:X,self.y:Y,self.const:self.CONST_LAMBDA})
                print(l2dist1, adv1, X, new_x1)
                
            else:
                
                if self.use_grad_op:

                    true_grads, los, l2s, los1, los2, scores, scores_x, nx, adv1 = self.sess.run([self.grad_op, self.loss, self.l2dist, self.loss1, self.loss2, self.output, self.output_x, self.new_x, self.adv], feed_dict={self.adv: real_adv, self.x:X, self.y:Y,self.const:self.CONST_LAMBDA})

                    true_grads[0][0][0:2,self.non_zeros:]=0

                    grad = true_grads[0].reshape(-1)

                else: 
                    
                    var = np.repeat(real_adv, self.batch_size*3, axis=0)
                    var_size = real_adv[0].size 
                    if support_mask is None:
                        update_indice = np.random.choice(a=list(range(self.non_zeros))+list(range(506,506+self.non_zeros)), size=1, replace=False,p=p) 
                    else:
                        update_indice = np.random.choice(a=support_mask, size=1, replace=False) 


                    grad = np.zeros(real_adv.reshape(-1).shape)
                    hh = self.h
                    while sum(grad)==0 and hh < 10: # to avoid gradient being zero
                        for i in range(self.batch_size):
                            var[self.batch_size * 1 + i].reshape(-1)[update_indice[0]] += hh
                            var[self.batch_size * 2 + i].reshape(-1)[update_indice[0]] -= hh

                        los, l2s, los_b, scores, nx, adv1 = self.sess.run([self.loss, self.l2dist, self.loss_batch, self.output, self.new_x, self.adv], feed_dict={self.adv: var, self.x:X3, self.y:Y3,self.const:LAMB})

                        grad = np.zeros(real_adv.reshape(-1).shape)

                        for i in range(self.batch_size):
                            grad[update_indice[0]] += los_b[self.batch_size * 1 + i]- los_b[self.batch_size * 2 + i]
                        grad[update_indice[0]] /= hh
                        if not self.adapt_h:
                            break
                        hh *= 2
                    if self.norm_gradient:
                        norm_grad = math.sqrt(sum(grad**2))
                        if norm_grad > 0:
                            grad /= norm_grad

                mt = self.beta1 * mt + (1 - self.beta1) * grad
                vt = self.beta2 * vt + (1 - self.beta2) * np.square(grad)
                corr = (math.sqrt(1 - self.beta2 ** epoch)) / (1 - self.beta1 ** epoch)

                m = real_adv.reshape(-1)
                m -= self.learning_rate * corr * (mt / (np.sqrt(vt) + self.epsilon))
                real_adv = m.reshape(real_adv.shape)

                if self.use_grad_op: 

                    if epoch == self.MAX_ITE:
                        print("attack failed!")
                        self.fail_count += 1
                        return 0
                    if sum((scores[0] == max(scores[0]))*Y[0])==0:
                        print("attack succeeded! with ite =", epoch)
                        self.success_count += 1
                        self.query_summary[t] = epoch
                        self.adv_summary[t] = adv1
                        return 1
                    

                else:

                    if epoch == self.MAX_ITE:
                        print("attack failed!")
                        if record_result:
                            if not self.adapt_lambda:
                                self.fail_count += 1
                            self.query_summary[t] = epoch
                            self.adv_summary[t] = real_adv
                            self.loss_lambda[t] = LAMB
                        return 0, vt, real_adv, math.sqrt(sum(sum(real_adv[0]**2))), epoch, LAMB
                    if sum((scores[0] == max(scores[0]))*Y[0])==0:
                        print("attack succeeded! with ite =", epoch)
                        if record_result:
                            self.success_count += 1
                            self.query_summary[t] = epoch
                            self.adv_summary[t] = real_adv
                            self.loss_lambda[t] = LAMB
                        return 1, vt, real_adv, math.sqrt(sum(sum(real_adv[0]**2))), epoch, LAMB
                    
BL4 = BlackboxL0(mode='blackbox', sess=sess, model=model, ckpt_path=ckpt_path, \
                 CONST_LAMBDA=10000, prob_thres=0.0, batch_size=1, learning_rate=1, h=1e-4, MAX_ITE=500, \
                 adapt_h=True, adapt_lambda=False)

for t in range(len(y_test)):  
    print("\n start attacking target", t, "...")
    success_flag,pred_score,pert,pert_norm,ite_num,lambda1=BL4.attack_L0(X_test, y_test, t)

attack_summary(BL4)