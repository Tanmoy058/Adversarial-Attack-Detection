from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable
from utils.layer_utils import AttentionLSTM
import tensorflow as tf
import math
import datetime
import numpy as np
from tqdm import tqdm
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from utils.generic_utils import load_dataset_at
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from utils.constants import MAX_NB_VARIABLES, MAX_TIMESTEPS_LIST

def current_milli_time():
    return int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000)

sess=tf.InteractiveSession()

#check utils/constants.py for dataset numbers. They are off by one so add 1 to them, cut_data is 0
DATASET_INDEX = 12

zeros_num = 10000

if DATASET_INDEX == 0:
    zeros_num = 1012
elif DATASET_INDEX == 6:
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

#model_2.save_weights(ckpt_path_2)

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

ckpt_path = ckpt_path_3
model = model_3

##################################################################################

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


times = []
##################################################################################

print("\n\n" + "Now beginning Blackbox Zoo results" + "\n\n")


success_count = 0
query_summary = {}
adv_summary = {}
fail_count = 0
invalid_count = 0
for t in range(X_test.shape[0]):
    
    #print("\n start attacking target", t, "...")
    
    mt = 0               # accumulator m_t in Adam
    vt = 0               # accumulator v_t in Adam
    beta1=0.9            # parameter beta_1 in Adam
    beta2=0.999          # parameter beta_2 in Adam
    learning_rate = 1e-1 # learning rate
    epsilon = 1e-8       # parameter epsilon in Adam
    h = 0.0001           # discretization constant when estimating numerical gradient
    batch_size = 1       # batch size
    MAX_ITE = 200        # maximum number of iterations


    real_adv = np.zeros([1, MAX_NB_VARIABLE, MAX_TIMESTEPS])   # initial adversarial perturbation, the trainable variable
    X = X_test[t:t+1]           # target sample X
    Y = y_test[t:t+1]           # target sample's lable Y
    
    non_zeros = max(sum(abs(X[0][0])>0), sum(abs(X[0][1])>0), 1)
    
    before = current_milli_time()
    pred_y = model.predict(X)
    after = current_milli_time()

    times.append((after-before)/len(X))

    if sum((max(pred_y[0])==pred_y[0]) * Y[0]) == 0:
        #print("not a valid target.")
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
                var_size = real_adv[0].size # (2, MAX_TIMESTEPS)

                update_indice = np.random.choice(list(range(non_zeros))+list(range(MAX_TIMESTEPS,MAX_TIMESTEPS+non_zeros)), 1, replace=False) 

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
                    #print("attack failed!")
                    fail_count += 1
                    break
                if sum((scores[0] == max(scores[0]))*Y[0])==0:
                    #print("attack successed! with ite =", epoch)
                    success_count += 1
                    query_summary[t] = epoch
                    adv_summary[t] = adv1
                    break

            else:
                if epoch == MAX_ITE:
                    #print("attack failed!")
                    fail_count += 1
                    break
                if sum((scores[0] == max(scores[0]))*Y[0])==0:
                    #print("attack successed! with ite =", epoch)
                    success_count += 1
                    query_summary[t] = epoch
                    adv_summary[t] = real_adv
                    break

if True == True:
    avg_l2_percent = []
    avg_l2_abs = []
    for success_index in query_summary.keys():
        if query_summary[success_index] != MAX_ITE and sum(sum(X_test[success_index]**2)) > 10:
            avg_l2_percent.append( math.sqrt( sum(sum(adv_summary[success_index][0]**2)) / sum(sum(X_test[success_index]**2)) ) )
            avg_l2_abs.append( math.sqrt( sum(sum(adv_summary[success_index][0]**2)) ) )

    if(len(avg_l2_percent) != 0):
        avg_l2 = sum(avg_l2_percent) / len(avg_l2_percent)
        avg_l2_ = sum(avg_l2_abs) / len(avg_l2_abs)
    else:
        avg_l2 = sum(avg_l2_percent)
        if(len(avg_l2_abs) != 0):
            avg_l2_ = sum(avg_l2_abs) / len(avg_l2_abs)
        else:
            avg_l2_ = sum(avg_l2_abs)
    a = []
    for i in query_summary.keys():
        if query_summary[i] != MAX_ITE:
            a.append(sum(sum(abs(adv_summary[i][0][:,:]) > 0)) /2/ max(sum(abs(X_test[i][0])>0), sum(abs(X_test[i][1])>0), 1))

    if(len(a) != 0):
        avg_l0 = sum(a) / len(a)
    else:
        avg_l0 = sum(a)
    if(success_count == 0):
        success_count = 1
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

    print("\n-------operation result-------\n")
    print("Successful times: {success_count}, \n\
    number of failures: {fail_count}, \n\
    Target model accuracy: {target_model_accuracy}, \n\
    attack Success rate: {success_rate}, \n\
    Average disturbancel2: {avg_l2_}, \n\
    Average disturbance l2 ratio: {avg_l2}, \n\
    Average disturbance l0 ratio: {avg_l0}, \n\
    Average number of iterations: {avg_ite}".format(**res_summary))

print("\n\n\nTIMES:")
print(times)
##################################################################################

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

    if fail_count == 0 and success_count == 0:
        success_count = 1

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

    print("\n-------operation result-------\n")
    print("Successful times: {success_count}, \n\
    invalid count: {invalid_count}, \n\
    number of failures: {fail_count}, \n\
    attack Success rate: {success_rate}, \n\
    Average disturbancel2: {avg_l2_}, \n\
    Average disturbance l2 ratio: {avg_l2}, \n\
    Average disturbance l0 ratio: {avg_l0}, \n\
    Average number of iterations: {avg_ite}".format(**res_summary))


##################################################################################

class BlackboxL2:
    def __init__(self, mode='blackbox', sess=sess, model=model, model_2=model_2, model_3=model_3, ckpt_path=ckpt_path, \
                 CONST_LAMBDA=10000, prob_thres=0.0, batch_size=1, learning_rate=2e-1, h=1e-4, MAX_ITE=2000, \
                 adapt_h=False, adapt_lambda=False, norm_gradient=False, option='vanilla'):

        self.success_count = 0
        self.query_summary = {}
        self.query_summary_funcal = {}
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


        self.option = option

        self.CONST_LAMBDA = CONST_LAMBDA
        self.SUCCESS_ATTACK_PROB_THRES = prob_thres
        
        self.x = tf.placeholder(tf.float32,[None, MAX_NB_VARIABLE, MAX_TIMESTEPS])
        self.y = tf.placeholder(tf.float32,[None, NB_CLASS])
        self.const = tf.placeholder(tf.float32)
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

        # In whitebox attack, Var adv is updated thru train_op, while in blackbox attack, adv is updated manually. 
        with tf.name_scope('attack'):
            if self.mode == 'whitebox':
                self.adv = tf.Variable(tf.zeros([1, MAX_NB_VARIABLE, MAX_TIMESTEPS]), name = "adv_pert")
            else:
                self.adv = tf.placeholder(tf.float32, [None, MAX_NB_VARIABLE, MAX_TIMESTEPS])

        # specify trainable variable
        all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        attack_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='attack')
        trainable_vars = tf.trainable_variables()
        for var in all_vars:
            if var not in attack_vars:
                trainable_vars.remove(var)

        # compute loss
        self.new_x = self.adv + self.x
        self.output_x = self.model(self.x)
        self.output = self.model(self.new_x)
        

        self.l2dist = tf.reduce_sum(tf.square(self.adv), [1,2])

        self.real = tf.reduce_sum(self.y * self.output, 1)
        self.fake = tf.reduce_max((1 - self.y) * self.output, 1)

        self.loss1 = self.const * tf.maximum(-self.SUCCESS_ATTACK_PROB_THRES, self.real - self.fake)
        self.loss2 = self.l2dist
        self.loss_batch = self.loss1 + self.loss2
        self.loss = tf.reduce_sum(self.loss_batch) # sum over all the batch samples

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # replace train_op with manual designed grad_op
        if self.use_train_op:
            self.train = optimizer.minimize(self.loss, var_list=trainable_vars)
        if self.use_grad_op:
            self.grad_op = tf.gradients(self.loss, self.adv)

        # initialize variables and load target model
        self.sess.run(tf.global_variables_initializer())
        self.model.load_weights(ckpt_path)
    
    def reset_summary(self):
        
        self.success_count = 0
        self.query_summary = {}
        self.adv_summary = {}
        self.loss_lambda = {}
        self.fail_count = 0
        self.invalid_count = 0

    def attack(self, X, Y, t, weight=np.ones((MAX_NB_VARIABLE, MAX_TIMESTEPS)), option='vanilla'):
        option = self.option
        X = X[t:t+1]
        Y = Y[t:t+1]
        
        if not self.adapt_lambda:
            return self.attack_lambda(X, Y, self.CONST_LAMBDA, t, weight, option=option)


        flag = 0
        L = self.CONST_LAMBDA
        while flag==0 and L <= 1e6:
            print("attempt to attack with lambda =", L, ":")
            flag,r1,rMAX_NB_VARIABLE,r3,r4,_ = self.attack_lambda(X, Y, L, t, weight, option=option)
            L *= 10
        self.fail_count += 1-flag
        return flag,r1,r2,r3,r4,L/10
        
    def attack_lambda(self, X, Y, LAMB, t, weight, option):
        
        mt = np.zeros((zeros_num,))                          # accumulator m_t in Adam
        vt = np.zeros((zeros_num,))                          # accumulator v_t in Adam

        real_adv = np.zeros([1, MAX_NB_VARIABLE, MAX_TIMESTEPS])   # initial adversarial perturbation, the trainable variable
    #         X = X_test[t:t+1]           # target sample X
    #         Y = y_test[t:t+1]           # target sample's lable Y

        self.non_zeros = max(sum(abs(X[0][0])>0), sum(abs(X[0][1])>0), 1)
        
        # coordinate sample distribution given by MWU
        p = np.hstack([weight[0,:self.non_zeros][::-1], weight[1,:self.non_zeros][::-1]])
        p /= sum(p)
    #         print(p[:self.non_zeros])

        # check if (X, Y) is a valid target 
        pred_y = self.model.predict(X)
        if sum((max(pred_y[0])==pred_y[0]) * Y[0]) == 0:
            #print("not a valid target.")
            self.invalid_count += 1
            return -1, 0, real_adv, 0, 0, 0
        
        if self.mode == 'blackbox':
            X3 = np.repeat(X, 3, axis=0)
            Y3 = np.repeat(Y, 3, axis=0)

        # main loop for the optimization
        num_query = 0
        for epoch in range(1, self.MAX_ITE+1):

            if self.use_train_op: # apply train_op

                # whitebox attack

                self.sess.run(self.train, feed_dict={x:X,y:Y})
                adv1, output1, l2dist1, real1, fake1, loss1, new_x1 = self.sess.run([self.adv, self.output, self.l2dist, self.real, self.fake, self.loss, self.new_x], feed_dict={self.x:X,self.y:Y,self.const:self.CONST_LAMBDA})
                print(l2dist1, adv1, X, new_x1)
                
            else: # apply self-implemented Adam

                # estimate gradient. 
                if self.use_grad_op: # For fake blackbox attack, just run grad_op.

                    # fake blackbox attack

                    true_grads, los, l2s, los1, los2, scores, scores_x, nx, adv1 = self.sess.run([self.grad_op, self.loss, self.l2dist, self.loss1, self.loss2, self.output, self.output_x, self.new_x, self.adv], feed_dict={self.adv: real_adv, self.x:X, self.y:Y,self.const:self.CONST_LAMBDA})

                    # clip the gradient to non-zero coordinates
                    true_grads[0][0][0:2,self.non_zeros:]=0

                    grad = true_grads[0].reshape(-1)

                else: # For blackbox attack, apply 1-order discretization

                    # blackbox attack   
                    var = np.repeat(real_adv, self.batch_size*3, axis=0)
                    var_size = real_adv[0].size # (2, MAX_TIMESTEPS)
                    # randomly choose a coordinate to compute partial gradient
    #                 update_indice = np.random.choice(var_size, 1, replace=True) 
        
        
                    # clip gradient
                    
                    # MWU for inner loop sampling
                    a = list(range(self.non_zeros))+list(range(MAX_TIMESTEPS,MAX_TIMESTEPS+self.non_zeros))
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
                    
                    # compute coordinate-perturbed input as a batch of size [batch_size*3, 2, MAX_TIMESTEPS]
                    # var = [X; X+h; X-h], X.size = [batch_size, 2, MAX_TIMESTEPS]
                    grad = np.zeros(real_adv.reshape(-1).shape)
                    hh = self.h
                    while sum(grad)==0 and hh < 20: # to avoid gradient being zero
                        for i in range(self.batch_size):
                            var[self.batch_size * 1 + i].reshape(-1)[update_indice[0]] += hh
                            var[self.batch_size * 2 + i].reshape(-1)[update_indice[0]] -= hh

                        los, l2s, los_b, scores, nx, adv1 = self.sess.run([self.loss, self.l2dist, self.loss_batch, self.output, self.new_x, self.adv], feed_dict={self.adv: var, self.x:X3, self.y:Y3,self.const:LAMB})
                        
                        num_query += 1
                        
                        grad = np.zeros(real_adv.reshape(-1).shape)

                        # grad(x) = [loss(X+he)-loss(X-he)] / (2h) 
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

                # Adam update
                mt = self.beta1 * mt + (1 - self.beta1) * grad
                vt = self.beta2 * vt + (1 - self.beta2) * np.square(grad)
                
                corr = (math.sqrt(1 - self.beta2 ** epoch)) / (1 - self.beta1 ** epoch)

                m = real_adv.reshape(-1)
                m -= self.learning_rate * corr * (mt / (np.sqrt(vt) + self.epsilon))
                real_adv = m.reshape(real_adv.shape)

                if self.use_grad_op: 
        #             print(los1, los2, scores, scores_x)
                    if epoch == self.MAX_ITE:
                        #print("attack failed!")
                        self.fail_count += 1
                        return 0
                    if sum((scores[0] == max(scores[0]))*Y[0])==0:
                        #print("attack succeeded! with ite =", epoch, "with query =", num_query)
                        self.success_count += 1
                        self.query_summary[t] = epoch
                        self.query_summary_funcal[t] = num_query
                        self.adv_summary[t] = adv1
                        return 1
                    

                else:
    #                 print(scores[0])
                    if epoch == self.MAX_ITE:
                        #print("attack failed!")
                        if not self.adapt_lambda:
                            self.fail_count += 1
                        self.query_summary[t] = epoch
                        self.query_summary_funcal[t] = num_query
                        self.adv_summary[t] = real_adv
                        self.loss_lambda[t] = LAMB
                        return 0, scores, real_adv, math.sqrt(sum(sum(real_adv[0]**2))), epoch, LAMB
                    if sum((scores[0] == max(scores[0]))*Y[0])==0:
                        #print("attack succeeded! with ite =", epoch, "with query =", num_query)
                        self.success_count += 1 
                        self.query_summary[t] = epoch
                        self.query_summary_funcal[t] = num_query
                        self.adv_summary[t] = real_adv
                        self.loss_lambda[t] = LAMB
                        return 1, scores, real_adv, math.sqrt(sum(sum(real_adv[0]**2))), epoch, LAMB

##################################################################################

class BlackboxL0:
    def __init__(self, mode='blackbox', sess=sess, model=model, ckpt_path=ckpt_path, \
                 CONST_LAMBDA=10000, prob_thres=0.0, batch_size=10, learning_rate=2e-1, h=1e-4, MAX_ITE=2000, \
                 adapt_h=False, adapt_lambda=False, norm_gradient=False, option='vanilla'):

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

        self.option = option

        
        self.CONST_LAMBDA = CONST_LAMBDA
        self.SUCCESS_ATTACK_PROB_THRES = prob_thres
        
        self.x = tf.placeholder(tf.float32,[None, MAX_NB_VARIABLE, MAX_TIMESTEPS])
        self.y = tf.placeholder(tf.float32,[None, NB_CLASS])
        self.const = tf.placeholder(tf.float32)
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

        # In whitebox attack, Var adv is updated thru train_op, while in blackbox attack, adv is updated manually. 
        with tf.name_scope('attack'):
            if self.mode == 'whitebox':
                self.adv = tf.Variable(tf.zeros([1, MAX_NB_VARIABLE, MAX_TIMESTEPS]), name = "adv_pert")
            else:
                self.adv = tf.placeholder(tf.float32, [None, MAX_NB_VARIABLE, MAX_TIMESTEPS])

        # specify trainable variable
        all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        attack_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='attack')
        trainable_vars = tf.trainable_variables()
        for var in all_vars:
            if var not in attack_vars:
                trainable_vars.remove(var)

        # compute loss
        self.new_x = self.adv + self.x
        self.output_x = self.model(self.x)
        self.output = self.model(self.new_x)

        self.l2dist = tf.reduce_sum(tf.square(self.adv), [1,2])

        self.real = tf.reduce_sum(self.y * self.output, 1)
        self.fake = tf.reduce_max((1 - self.y) * self.output, 1)

        self.loss1 = self.const * tf.maximum(-self.SUCCESS_ATTACK_PROB_THRES, self.real - self.fake)
        self.loss2 = self.l2dist
        self.loss_batch = self.loss1 + self.loss2
        self.loss = tf.reduce_sum(self.loss_batch) # sum over all the batch samples

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # replace train_op with manual designed grad_op
        if self.use_train_op:
            self.train = optimizer.minimize(self.loss, var_list=trainable_vars)
        if self.use_grad_op:
            self.grad_op = tf.gradients(self.loss, self.adv)

        # initialize variables and load target model
        self.sess.run(tf.global_variables_initializer())
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
    
    def attack_L0(self, X, Y, t, weight=np.ones((MAX_NB_VARIABLE, MAX_TIMESTEPS))):
        
        flag, scores, pert, l2dist, epoch, valid_lbd = self.attack(X, Y, t, np.zeros([1, MAX_NB_VARIABLE, MAX_TIMESTEPS]), weight=weight)
        
        self.vt = scores # record vt
        
        if flag < 1: # L2 attack failed then
            return flag, scores, pert, l2dist, epoch, valid_lbd
        # L2 attack succeed then
        
        l2_anchor = l2dist
        support_mask = [0, 1]
        while l2dist < l2_anchor * 1.1 and len(support_mask) > 1:
            support_mask = [i for i in list(range(self.non_zeros))+list(range(MAX_TIMESTEPS,MAX_TIMESTEPS+self.non_zeros)) if abs(pert.reshape(-1)[i]) > 0]
            
            shrink_indice = self.yield_shrink_indice(pert, method='vt-max')
            print(shrink_indice)
    
            if shrink_indice in support_mask:
                support_mask.remove(shrink_indice)
                pert_shrink = pert.copy()
                pert_shrink.reshape(-1)[shrink_indice] = 0
            else:
                #print("unexpected shrink indice detected.")
                return 0,0,0,0,0,0
            
            #print("L2 -> L0 squeeze: L2=", l2dist, ", L0=", len(support_mask), ", epoch=", epoch)

            flag_1, scores_1, pert_1, l2dist_1, epoch_1, valid_lbd_1 = self.attack_lambda(X, Y, t, valid_lbd, pert_shrink, support_mask, record_result=False, weight=weight)
            
            if flag_1 < 1: # L2 attack failed then
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
        
        

    def attack(self, X, Y, t, init_pert=np.zeros([1, MAX_NB_VARIABLE, MAX_TIMESTEPS]), weight=np.ones((MAX_NB_VARIABLE, MAX_TIMESTEPS))):
        
        
        if not self.adapt_lambda:
            return self.attack_lambda(X, Y, t, self.CONST_LAMBDA, init_pert, weight=weight)

        flag = 0
        L = self.CONST_LAMBDA
        while flag==0 and L <= 1e6:
            #print("attempt to attack with lambda =", L, ":")
            flag,r1,r2,r3,r4,_ = self.attack_lambda(X, Y, t, L, init_pert, weight=weight)
            L *= 10
        self.fail_count += 1-flag
        return flag,r1,r2,r3,r4,L/10
        
    def attack_lambda(self, X, Y, t, LAMB, init_pert=np.zeros([1, MAX_NB_VARIABLE, MAX_TIMESTEPS]), support_mask=None, record_result=True, weight=np.ones((MAX_NB_VARIABLE, MAX_TIMESTEPS))):
        
        X = X[t:t+1]
        Y = Y[t:t+1]
        
        mt = np.zeros((zeros_num,))                          # accumulator m_t in Adam
        vt = np.zeros((zeros_num,))     

        real_adv = init_pert   # initial adversarial perturbation, the trainable variable
        
        self.non_zeros = max(sum(abs(X[0][0])>0), sum(abs(X[0][1])>0), 1)
        
        # coordinate sample distribution given by MWU
        p = np.hstack([weight[0,:self.non_zeros][::-1], weight[1,:self.non_zeros][::-1]])
        p /= sum(p)

        # check if (X, Y) is a valid target 
        pred_y = self.model.predict(X)
        if sum((max(pred_y[0])==pred_y[0]) * Y[0]) == 0:
            #print("not a valid target.")
            self.invalid_count += 1
            return -1, 0, real_adv, 0, 0, 0
        
        if self.mode == 'blackbox':
            X3 = np.repeat(X, 3, axis=0)
            Y3 = np.repeat(Y, 3, axis=0)

        # main loop for the optimization
        for epoch in range(1, self.MAX_ITE+1):

            if self.use_train_op: # apply train_op

                # whitebox attack

                self.sess.run(self.train, feed_dict={x:X,y:Y})
                adv1, output1, l2dist1, real1, fake1, loss1, new_x1 = self.sess.run([self.adv, self.output, self.l2dist, self.real, self.fake, self.loss, self.new_x], feed_dict={self.x:X,self.y:Y,self.const:self.CONST_LAMBDA})
                print(l2dist1, adv1, X, new_x1)
                
            else: # apply self-implemented Adam

                # estimate gradient. 
                if self.use_grad_op: # For fake blackbox attack, just run grad_op.

                    # fake blackbox attack

                    true_grads, los, l2s, los1, los2, scores, scores_x, nx, adv1 = self.sess.run([self.grad_op, self.loss, self.l2dist, self.loss1, self.loss2, self.output, self.output_x, self.new_x, self.adv], feed_dict={self.adv: real_adv, self.x:X, self.y:Y,self.const:self.CONST_LAMBDA})

                    # clip the gradient to non-zero coordinates
                    true_grads[0][0][0:2,self.non_zeros:]=0

                    grad = true_grads[0].reshape(-1)

                else: # For blackbox attack, apply 1-order discretization

                    # blackbox attack   
                    var = np.repeat(real_adv, self.batch_size*3, axis=0)
                    var_size = real_adv[0].size # (MAX_NB_VARIABLE, MAX_TIMESTEPS)
                    # randomly choose a coordinate to compute partial gradient
    #                 update_indice = np.random.choice(var_size, 1, replace=True) 
                    # clip gradient
                    if support_mask is None or len(support_mask) == 0:
                        update_indice = np.random.choice(a=list(range(self.non_zeros))+list(range(MAX_TIMESTEPS,MAX_TIMESTEPS+self.non_zeros)), size=1, replace=False, p=p) 
                    else:
                        update_indice = np.random.choice(a=support_mask, size=1, replace=False) 

                    # compute coordinate-perturbed input as a batch of size [batch_size*3, MAX_NB_VARIABLE, MAX_TIMESTEPS]
                    # var = [X; X+h; X-h], X.size = [batch_size, MAX_NB_VARIABLE, MAX_TIMESTEPS]
                    grad = np.zeros(real_adv.reshape(-1).shape)
                    hh = self.h
                    while sum(grad)==0 and hh < 10: # to avoid gradient being zero
                        for i in range(self.batch_size):
                            var[self.batch_size * 1 + i].reshape(-1)[update_indice[0]] += hh
                            var[self.batch_size * 2 + i].reshape(-1)[update_indice[0]] -= hh

                        los, l2s, los_b, scores, nx, adv1 = self.sess.run([self.loss, self.l2dist, self.loss_batch, self.output, self.new_x, self.adv], feed_dict={self.adv: var, self.x:X3, self.y:Y3,self.const:LAMB})

                        grad = np.zeros(real_adv.reshape(-1).shape)

                        # grad(x) = [loss(X+he)-loss(X-he)] / (2h) 
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

                # Adam update
                mt = self.beta1 * mt + (1 - self.beta1) * grad
                vt = self.beta2 * vt + (1 - self.beta2) * np.square(grad)
                corr = (math.sqrt(1 - self.beta2 ** epoch)) / (1 - self.beta1 ** epoch)

                m = real_adv.reshape(-1)
                m -= self.learning_rate * corr * (mt / (np.sqrt(vt) + self.epsilon))
                real_adv = m.reshape(real_adv.shape)

                if self.use_grad_op: 
        #             print(los1, los2, scores, scores_x)
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
    #                 print(scores[0])
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


class BlackboxL0MWU:
    def __init__(self, mode='blackbox', sess=sess, model=model, ckpt_path=ckpt_path, \
                 CONST_LAMBDA=10000, prob_thres=0.0, batch_size=10, learning_rate=2e-1, h=1e-4, MAX_ITE=2000, \
                 adapt_h=False, adapt_lambda=False, norm_gradient=False, option='vanilla'):

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

        self.option = option

        
        self.CONST_LAMBDA = CONST_LAMBDA
        self.SUCCESS_ATTACK_PROB_THRES = prob_thres
        
        self.x = tf.placeholder(tf.float32,[None, MAX_NB_VARIABLE, MAX_TIMESTEPS])
        self.y = tf.placeholder(tf.float32,[None, NB_CLASS])
        self.const = tf.placeholder(tf.float32)
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

        # In whitebox attack, Var adv is updated thru train_op, while in blackbox attack, adv is updated manually. 
        with tf.name_scope('attack'):
            if self.mode == 'whitebox':
                self.adv = tf.Variable(tf.zeros([1, MAX_NB_VARIABLE, MAX_TIMESTEPS]), name = "adv_pert")
            else:
                self.adv = tf.placeholder(tf.float32, [None, MAX_NB_VARIABLE, MAX_TIMESTEPS])

        # specify trainable variable
        all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        attack_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='attack')
        trainable_vars = tf.trainable_variables()
        for var in all_vars:
            if var not in attack_vars:
                trainable_vars.remove(var)

        # compute loss
        self.new_x = self.adv + self.x
        self.output_x = self.model(self.x)
        self.output = self.model(self.new_x)

        self.l2dist = tf.reduce_sum(tf.square(self.adv), [1,2])

        self.real = tf.reduce_sum(self.y * self.output, 1)
        self.fake = tf.reduce_max((1 - self.y) * self.output, 1)

        self.loss1 = self.const * tf.maximum(-self.SUCCESS_ATTACK_PROB_THRES, self.real - self.fake)
        self.loss2 = self.l2dist
        self.loss_batch = self.loss1 + self.loss2
        self.loss = tf.reduce_sum(self.loss_batch) # sum over all the batch samples

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # replace train_op with manual designed grad_op
        if self.use_train_op:
            self.train = optimizer.minimize(self.loss, var_list=trainable_vars)
        if self.use_grad_op:
            self.grad_op = tf.gradients(self.loss, self.adv)

        # initialize variables and load target model
        self.sess.run(tf.global_variables_initializer())
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
    
    def attack_L0(self, X, Y, t, weight=np.ones((MAX_NB_VARIABLE, MAX_TIMESTEPS))):
        
        flag, scores, pert, l2dist, epoch, valid_lbd = self.attack(X, Y, t, np.zeros([1, MAX_NB_VARIABLE, MAX_TIMESTEPS]), weight=weight)
        
        self.vt = scores # record vt
        
        if flag < 1: # L2 attack failed then
            return flag, scores, pert, l2dist, epoch, valid_lbd
        # L2 attack succeed then
        
        l2_anchor = l2dist
        support_mask = [0, 1]
        while l2dist < l2_anchor * 1.1 and len(support_mask) > 1:
            support_mask = [i for i in list(range(self.non_zeros))+list(range(MAX_TIMESTEPS,MAX_TIMESTEPS+self.non_zeros)) if abs(pert.reshape(-1)[i]) > 0]
            
            shrink_indice = self.yield_shrink_indice(pert, method='vt-max')
            #print(shrink_indice)

            
            if shrink_indice in support_mask:
                support_mask.remove(shrink_indice)
                pert_shrink = pert.copy()
                pert_shrink.reshape(-1)[shrink_indice] = 0
            else:
                #print("unexpected shrink indice detected.")
                return 0,0,0,0,0,0
            
            #print("L2 -> L0 squeeze: L2=", l2dist, ", L0=", len(support_mask), ", epoch=", epoch)

            flag_1, scores_1, pert_1, l2dist_1, epoch_1, valid_lbd_1 = self.attack_lambda(X, Y, t, valid_lbd, pert_shrink, support_mask, record_result=False, weight=weight)
            
            if flag_1 < 1: # L2 attack failed then
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
        
        

    def attack(self, X, Y, t, init_pert=np.zeros([1, MAX_NB_VARIABLE, MAX_TIMESTEPS]), weight=np.ones((MAX_NB_VARIABLE, MAX_TIMESTEPS))):
        
        
        if not self.adapt_lambda:
            return self.attack_lambda(X, Y, t, self.CONST_LAMBDA, init_pert, weight=weight)

        flag = 0
        L = self.CONST_LAMBDA
        while flag==0 and L <= 1e6:
            #print("attempt to attack with lambda =", L, ":")
            flag,r1,r2,r3,r4,_ = self.attack_lambda(X, Y, t, L, init_pert, weight=weight)
            L *= 10
        self.fail_count += 1-flag
        return flag,r1,r2,r3,r4,L/10
        
    def attack_lambda(self, X, Y, t, LAMB, init_pert=np.zeros([1, MAX_NB_VARIABLE, MAX_TIMESTEPS]), support_mask=None, record_result=True, weight=np.ones((MAX_NB_VARIABLE, MAX_TIMESTEPS)), option='mwu-outer'):
        
        X = X[t:t+1]
        Y = Y[t:t+1]
        
        mt = np.zeros((zeros_num,))                          # accumulator m_t in Adam
        vt = np.zeros((zeros_num,))     

        real_adv = init_pert   # initial adversarial perturbation, the trainable variable
        
        self.non_zeros = max(sum(abs(X[0][0])>0), sum(abs(X[0][1])>0), 1)
        
        # coordinate sample distribution given by MWU
        p = np.hstack([weight[0,:self.non_zeros][::-1], weight[1,:self.non_zeros][::-1]])
        p /= sum(p)

        # check if (X, Y) is a valid target 
        pred_y = self.model.predict(X)
        if sum((max(pred_y[0])==pred_y[0]) * Y[0]) == 0:
            #print("not a valid target.")
            self.invalid_count += 1
            return -1, 0, real_adv, 0, 0, 0
        
        if self.mode == 'blackbox':
            X3 = np.repeat(X, 3, axis=0)
            Y3 = np.repeat(Y, 3, axis=0)

        # main loop for the optimization
        for epoch in range(1, self.MAX_ITE+1):

            if self.use_train_op: # apply train_op

                # whitebox attack

                self.sess.run(self.train, feed_dict={x:X,y:Y})
                adv1, output1, l2dist1, real1, fake1, loss1, new_x1 = self.sess.run([self.adv, self.output, self.l2dist, self.real, self.fake, self.loss, self.new_x], feed_dict={self.x:X,self.y:Y,self.const:self.CONST_LAMBDA})
                print(l2dist1, adv1, X, new_x1)
                
            else: # apply self-implemented Adam

                # estimate gradient. 
                if self.use_grad_op: # For fake blackbox attack, just run grad_op.

                    # fake blackbox attack

                    true_grads, los, l2s, los1, los2, scores, scores_x, nx, adv1 = self.sess.run([self.grad_op, self.loss, self.l2dist, self.loss1, self.loss2, self.output, self.output_x, self.new_x, self.adv], feed_dict={self.adv: real_adv, self.x:X, self.y:Y,self.const:self.CONST_LAMBDA})

                    # clip the gradient to non-zero coordinates
                    true_grads[0][0][0:2,self.non_zeros:]=0

                    grad = true_grads[0].reshape(-1)

                else: # For blackbox attack, apply 1-order discretization

                    # blackbox attack   
                    var = np.repeat(real_adv, self.batch_size*3, axis=0)
                    var_size = real_adv[0].size # (MAX_NB_VARIABLE, MAX_TIMESTEPS)
                    # randomly choose a coordinate to compute partial gradient
    #                 update_indice = np.random.choice(var_size, 1, replace=True) 
                    # clip gradient
                    a = list(range(self.non_zeros))+list(range(MAX_TIMESTEPS,MAX_TIMESTEPS+self.non_zeros))
                    ep = (1/(1+np.exp(np.sqrt(vt[a]) * -0.1)))**4
                    p1 = ep / sum(ep)
                    p2 = p1
                    update_indice = np.random.choice(a=a, size=1, replace=False)
                    if option=='vanilla':
                        update_indice = np.random.choice(a=a, size=1, replace=False) 
                    if option=='mwu-inner':
                        update_indice = np.random.choice(a=a, size=1, replace=False, p=p1) 
                    if option=='mwu-outer':
                        update_indice = np.random.choice(a=a, size=1, replace=False, p=p) 
                    if option=='mwu-mix':
                        p2 = p*p1
                        update_indice = np.random.choice(a=a, size=1, replace=False, p=p2/sum(p2))

                    # compute coordinate-perturbed input as a batch of size [batch_size*3, MAX_NB_VARIABLE, MAX_TIMESTEPS]
                    # var = [X; X+h; X-h], X.size = [batch_size, MAX_NB_VARIABLE, MAX_TIMESTEPS]
                    grad = np.zeros(real_adv.reshape(-1).shape)
                    hh = self.h
                    while sum(grad)==0 and hh < 10: # to avoid gradient being zero
                        for i in range(self.batch_size):
                            var[self.batch_size * 1 + i].reshape(-1)[update_indice[0]] += hh
                            var[self.batch_size * 2 + i].reshape(-1)[update_indice[0]] -= hh

                        los, l2s, los_b, scores, nx, adv1 = self.sess.run([self.loss, self.l2dist, self.loss_batch, self.output, self.new_x, self.adv], feed_dict={self.adv: var, self.x:X3, self.y:Y3,self.const:LAMB})

                        grad = np.zeros(real_adv.reshape(-1).shape)

                        # grad(x) = [loss(X+he)-loss(X-he)] / (2h) 
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

                # Adam update
                mt = self.beta1 * mt + (1 - self.beta1) * grad
                vt = self.beta2 * vt + (1 - self.beta2) * np.square(grad)
                corr = (math.sqrt(1 - self.beta2 ** epoch)) / (1 - self.beta1 ** epoch)

                m = real_adv.reshape(-1)
                m -= self.learning_rate * corr * (mt / (np.sqrt(vt) + self.epsilon))
                real_adv = m.reshape(real_adv.shape)

                if self.use_grad_op: 
        #             print(los1, los2, scores, scores_x)
                    if epoch == self.MAX_ITE:
                        #print("attack failed!")
                        self.fail_count += 1
                        return 0
                    if sum((scores[0] == max(scores[0]))*Y[0])==0:
                        #print("attack succeeded! with ite =", epoch)
                        self.success_count += 1
                        self.query_summary[t] = epoch
                        self.adv_summary[t] = adv1
                        return 1
                    

                else:
    #                 print(scores[0])
                    if epoch == self.MAX_ITE:
                        #print("attack failed!")
                        if record_result:
                            if not self.adapt_lambda:
                                self.fail_count += 1
                            self.query_summary[t] = epoch
                            self.adv_summary[t] = real_adv
                            self.loss_lambda[t] = LAMB
                        return 0, vt, real_adv, math.sqrt(sum(sum(real_adv[0]**2))), epoch, LAMB
                    if sum((scores[0] == max(scores[0]))*Y[0])==0:
                        #print("attack succeeded! with ite =", epoch)
                        if record_result:
                            self.success_count += 1
                            self.query_summary[t] = epoch
                            self.adv_summary[t] = real_adv
                            self.loss_lambda[t] = LAMB
                        return 1, vt, real_adv, math.sqrt(sum(sum(real_adv[0]**2))), epoch, LAMB

###################################################################################
'''
print("\n\n" + "Now beginning Blackbox L0 results" + "\n\n")

L0_attack = BlackboxL0(mode='blackbox', sess=sess, model=model, ckpt_path=ckpt_path, \
                 CONST_LAMBDA=10000, prob_thres=0.0, batch_size=1, learning_rate=2e-1, h=1e-4, MAX_ITE=500, \
                 adapt_h=False, adapt_lambda=False)

for t in range(len(y_test)):  
    #print("\n start attacking target", t, "...")
    success_flag,pred_score,pert,pert_norm,ite_num,lambda1=L0_attack.attack_L0(X_test, y_test, t)


attack_summary(L0_attack)
'''

###################################################################################

'''
print("\n\n" + "Now beginning Blackbox AdaZoo results" + "\n\n")

AdaZoo = BlackboxL2(mode='blackbox', sess=sess, model=model, ckpt_path=ckpt_path, \
                 CONST_LAMBDA=10000, prob_thres=0.0, batch_size=1, learning_rate=2e-1, h=1e-4, MAX_ITE=500, \
                 adapt_h=True, adapt_lambda=False)

for t in range(len(y_test)):  
    #print("\n start attacking target", t, "...")
    success_flag,pred_score,pert,pert_norm,ite_num,lambda1=AdaZoo.attack(X_test, y_test, t)

attack_summary(AdaZoo)



print("\n\n" + "Now beginning Blackbox L0-MWU results" + "\n\n")

MWU_L0_attack = BlackboxL0MWU(mode='blackbox', sess=sess, model=model, ckpt_path=ckpt_path, \
                 CONST_LAMBDA=10000, prob_thres=0.0, batch_size=1, learning_rate=2e-1, h=1e-4, MAX_ITE=500, \
                 adapt_h=False, adapt_lambda=False)

for t in range(len(y_test)):  
    #print("\n start attacking target", t, "...")
    success_flag,pred_score,pert,pert_norm,ite_num,lambda1=MWU_L0_attack.attack_L0(X_test, y_test, t)


attack_summary(MWU_L0_attack)
'''
###################################################################################



'''

h_val = .0001
print("\n\n" + "Now beginning Blackbox L2 results" + "\n\n")


L2_attack = BlackboxL2(mode='blackbox', sess=sess, model=model, ckpt_path=ckpt_path, \
                 CONST_LAMBDA=10000, prob_thres=0.0, batch_size=1, learning_rate=2e-1, h=h_val, MAX_ITE=500, \
                 adapt_h=False, adapt_lambda=False)

weight = np.ones((MAX_NB_VARIABLE, MAX_TIMESTEPS))

for t in range(len(y_test)):
    #print("\n start attacking target", t, "...")
    success_flag,pred_score,pert,pert_norm,_,_=L2_attack.attack(X_test, y_test, t, weight, option='vanilla')

attack_summary(L2_attack)


'''

