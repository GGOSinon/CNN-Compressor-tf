import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
from ImageFunctions import *

# 40 x 40 images
Data = np.load("../data/data_240.npy")
Data_val = np.load("../data/data_240_val.npy") 
training_iters = Data.shape[0]
n_valid = Data_val.shape[0]

#training_iters = 32
#n_valid = 3
batch_size = 8
display_step = 10
tot_step = training_iters/batch_size
n_repeat = 1
n_decay = 50
decay_step = (tot_step//n_decay)
qf = 10
var_list = []

#print(decay_step)
#Learning rate : Best 00005
c_learning_start = 0.0005
g_learning_start = 0.005
i_learning_start = 0.01
r_learning_start = 0.01
c_learning_end = 0.00005
g_learning_end = 0.00005
i_learning_end = 0.0001
r_learning_end = 0.0001

com_train = 1
gen_train = 1
img_train = 1
cor_train = 1

c_decay_rate = pow(c_learning_end/c_learning_start,1./(n_repeat*n_decay*com_train))
g_decay_rate = pow(g_learning_end/g_learning_start,1./(n_repeat*n_decay*gen_train))
i_decay_rate = pow(i_learning_end/i_learning_start,1./(n_repeat*n_decay*img_train))
r_decay_rate = pow(r_learning_end/r_learning_start,1./(n_repeat*n_decay*cor_train))

glob_step_c = tf.Variable(0, trainable = False)
glob_step_g = tf.Variable(0, trainable = False)
glob_step_i = tf.Variable(0, trainable = False)
glob_step_r = tf.Variable(0, trainable = False)

c_learning_rate = tf.train.exponential_decay(c_learning_start, glob_step_c, decay_step, c_decay_rate, staircase=True)
g_learning_rate = tf.train.exponential_decay(g_learning_start, glob_step_g, decay_step, g_decay_rate, staircase=True)
i_learning_rate = tf.train.exponential_decay(i_learning_start, glob_step_i, decay_step, i_decay_rate, staircase=True)
r_learning_rate = tf.train.exponential_decay(r_learning_start, glob_step_r, decay_step, r_decay_rate, staircase=True)

#c_learning_rate = c_learning_start
#g_learning_rate = g_learning_start
n_prob = 20
h_prob, w_prob = 3, 3
c, h, w = Data.shape[3], Data.shape[1], Data.shape[2]
c_val, h_val, w_val = Data_val.shape[3], Data_val.shape[1], Data_val.shape[2]
compress_rate = 4
hh, ww = h/compress_rate, w/compress_rate

gpu_device = ['/device:GPU:0', '/device:GPU:1']
img_input = tf.placeholder(tf.float32, [None, h, w, c])
img_ans = tf.placeholder(tf.float32, [None, h, w, c]) 
img_input_cor = tf.placeholder(tf.float32, [None, h, w, c])
img_input_gen = tf.placeholder(tf.float32, [None, h, w, c])

def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)

def conv2d(x, W, b, stride = 1, act_func = 'LReLU', use_bn = True):
    strides = [1, stride, stride, 1]
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides, padding='SAME')
    x = tf.nn.bias_add(x, b)
    if act_func == 'LReLU': x = leaky_relu(x)
    if act_func == 'ReLU': x = tf.nn.relu(x)
    if act_func == 'TanH': x = tf.nn.tanh(x)
    if act_func == 'PReLU': x = leaky_relu(x)#tf.keras.layers.PReLU(x)
    if act_func == 'Sigmoid': x = tf.nn.sigmoid(x)
    if act_func == 'Softmax':
        x = tf.reshape(x, (-1, h*w))
        x = tf.nn.softmax(x, axis=1)
        x = tf.reshape(x, (-1, h, w, 1))
    if act_func == 'None': pass
    if use_bn: return slim.batch_norm(x)
    else: return x

def com_net(x, weights, biases):
    x = tf.reshape(x, (-1, h, w, c))
    #x_input = x
    x = conv2d(x, weights['wc1'], biases['bc1'], act_func = 'LReLU', use_bn = False)
    
    for i in range(2, 2):
    	name_w = 'wc'+str(i)
    	name_b = 'bc'+str(i)
    	x = conv2d(x, weights[name_w], biases[name_b], act_func = 'LReLU')   
    
    res = conv2d(x, weights['wcx'], biases['bcx'], act_func='Sigmoid', use_bn = False)
    return res

def gen_net(x, weights, biases):
    x = tf.reshape(x, (-1, h, w, c))
    x = conv2d(x, weights['wc1'], biases['bc1'], act_func = 'LReLU', use_bn = False) 
    for i in range(1, 8//2):
    	x_input = x
    	name_w = 'wc'+str(2*i)
    	name_b = 'bc'+str(2*i)
    	x_input = conv2d(x_input, weights[name_w], biases[name_b], act_func='LReLU')
    	name_w = 'wc'+str(2*i+1)
    	name_b = 'bc'+str(2*i+1)
    	x_input = conv2d(x_input, weights[name_w], biases[name_b], act_func='None')
    	x = leaky_relu(x_input + x)
    res = conv2d(x, weights['wcx'], biases['bcx'], act_func='TanH', use_bn = False)
    return res

def cor_net(x, weights, biases):
    x = tf.reshape(x, (-1, h, w, c))
    #return x
    x = conv2d(x, weights['wc1'], biases['bc1'], act_func = 'LReLU', use_bn = False)
    for i in range(2, 3):
    	name_w = 'wc'+str(i)
    	name_b = 'bc'+str(i)
    	x = conv2d(x, weights[name_w], biases[name_b], act_func = 'LReLU') 
    res = conv2d(x, weights['wcx'], biases['bcx'], act_func='TanH', use_bn = False)
    #fc1 = conv2d(conv3, weights['wd1'], biases['bd1'])
    #out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return res

def make_grad(s, e, name):
    var = tf.get_variable(name, [3, 3, s, e], initializer=tf.contrib.layers.xavier_initializer())
    var_list.append(var)
    return var

def make_bias(x, name):
    var = tf.get_variable(name, [x], initializer=tf.contrib.layers.xavier_initializer())
    var_list.append(var)
    return var

def make_dict(num_layer, num_filter, end_str, s_filter = 1, e_filter = 1):
    
    result = {}
    weights = {}
    biases = {}
    
    weights['wc1'] = make_grad(s_filter,num_filter,"w1"+end_str)
    for i in range(2, num_layer):
    	index = 'wc' + str(i)
    	name = 'w' + str(i) + end_str
    	weights[index] = make_grad(num_filter, num_filter, name)
    weights['wcx'] = make_grad(num_filter,e_filter,"wx"+end_str)
    
    biases['bc1'] = make_bias(num_filter,"b1"+end_str)
    for i in range(2, num_layer):
    	index = 'bc' + str(i)
    	#print(index)
    	name = 'b' + str(i) + end_str
    	biases[index] = make_bias(num_filter, name)
    biases['bcx'] = make_bias(e_filter,"bx"+end_str)
    
    result['weights'] = weights
    result['biases'] = biases
    return result

var_com = make_dict(5, 32, 'c', 1, 1)
print(var_com)
var_gen = make_dict(25, 32, 'g', 1, 1)
var_img = make_dict(25, 32, 'i', 1, 1)
var_cor = make_dict(25, 32, 'r', 1, 1)


print(Data.shape)
print(Data_val.shape)
global_step = 0

print(training_iters)
print(n_valid)

def valid_batch(prev, size):
	ans = []
	data = []
	for i in range(size):
		if i+prev>=n_valid: continue
		data.append(Data_val[i+prev])
	data = np.array(data)
	data = data.reshape(-1, h_val, w_val, c_val)
	return data, data

def next_batch(prev, size):
    ans = []
    data = []
    for i in range(size):
        if i+prev>=training_iters: continue
        data.append(Data[RandArr[i+prev]])
    data = np.array(data)
    data = data.reshape(-1, h, w, c)
    return data, data

# Define graph

# Com
img_com = com_net(img_input, var_com['weights'], var_com['biases'])
img_res = gen_net(img_com, var_gen['weights'],var_gen['biases'])

img_res_gen = gen_net(img_input_gen, var_gen['weights'], var_gen['biases'])

#img_final = img_res_gen + img_uscale
img_final = img_res_gen + img_input_gen

# Img
def normalize_delta(x, scale = 10):
    
    x = tf.reshape(x, (-1, h*w))
    x = tf.abs(x)
    '''
    S = tf.reduce_sum(x, axis=1, keep_dims=True)
    x /= S
    x = tf.reshape(x, (-1, h, w))
    '''
    #min_x = tf.reduce_min(x, axis = 1, keep_dims=True)
    max_x = tf.reduce_max(x, axis = 1, keep_dims=True)
    res = x / max_x
    res = tf.nn.softmax(scale * res, axis=1)
    res = tf.reshape(res, (-1, h, w))
    return res

# Cor
def cor_compress(img, img_ans):
    sz_prob = img.shape[0]
    img_inc = np.zeros([sz_prob, n_prob], dtype=np.int32)
    img_val = np.zeros([sz_prob, n_prob], dtype=np.float32)
    p_img = np.absolute(img - img_ans)
    for i in range(sz_prob):
        p_img_flatten = np.reshape(p_img[i], [-1])
        img_flatten = np.reshape(img_ans[i] - img[i], [-1])
        incides = np.argpartition(p_img_flatten,-n_prob)[-n_prob:]
        for j in range(n_prob):
            inc = incides[j]
            img_inc[i][j] = inc
            img_val[i][j] = img_flatten[inc]
    #img_res = np.array(img_res)
    #img_input_cor = tf.convert_to_tensor(img_input_cor_np)
    return img_inc, img_val

def cor_decompress(img_inc, img_val):
    sz_img = img_inc.shape[0]
    img_decom = np.zeros([sz_img, h*w*c], dtype=np.float32)
    for i in range(sz_img):
        for j in range(n_prob):
            inc, val = img_inc[i][j], img_val[i][j]
            img_decom[i][inc] = val
    img_decom = np.reshape(img_decom, [-1, h, w, c])
    return img_decom

img_cor = cor_net(img_input_cor,var_cor['weights'],var_cor['biases'])

img_real_final = img_final + img_cor

com_loss = tf.losses.mean_squared_error(img_ans, img_com + img_res)
gen_loss = tf.losses.mean_squared_error(img_ans, img_final)
cor_loss = tf.losses.mean_squared_error(img_ans - img_final, img_cor)


c_opt = tf.train.AdamOptimizer(learning_rate=c_learning_rate, epsilon = 1e-4)
g_opt = tf.train.AdamOptimizer(learning_rate=g_learning_rate, epsilon = 1e-4)
r_opt = tf.train.AdamOptimizer(learning_rate=r_learning_rate, epsilon = 1e-4)

def create_op(opt, loss, var_list, glob_step):
     
    grads, var = zip(*opt.compute_gradients(loss, var_list = var_list))
    #grads = [None if grad is None else tf.clip_by_value(grad, -1., 1.0) for grad in grads]
    grads = [None if grad is None else tf.clip_by_norm(grad, 0.5) for grad in grads]
    return opt.apply_gradients(zip(grads, var), global_step = glob_step)
     
    '''
    gvs = opt.compute_gradients(loss, var_list=var_list)
    capped_gvs = [None if grad is None else (tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    return opt.apply_gradients(capped_gvs, global_step = glob_step)
    m
    ''' 
'''
def get_reg(var_list, name_weight):
    weight_list = {}
    for name, weights in var_list:
        weight = tf.reduce_mean(weights)
        weight_list[name] = weight
    if name_weight=='com':
        return weight_list[0] + weight_list[1]
'''
#com_reg_loss = get_reg(var_com['weights'])
#com_loss += com_reg_loss

#com_op = create_op(c_opt, com_loss, var_com, glob_step_c)
#gen_op = create_op(g_opt, gen_loss, var_gen, glob_step_g)
#img_op = create_op(i_opt, img_loss, var_img, glob_step_i)
cor_op = create_op(r_opt, cor_loss, var_cor, glob_step_r)

# Initializing the variables
init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
max_acc = 0.

p_acc = []
Log = open('Log_r.txt', 'w')
Log.close()
saver = tf.train.Saver(var_list = var_list, max_to_keep = None)
#saver = tf.train.Saver()

turn = 'gen'
min_loss = 1.

with sess:
    sess.run(init)
    saver.restore(sess, "./model/best/cg-model.ckpt-qf=10-best")
    #sess.run(init)
    # Keep training until reach max iterations
    for i in range(n_repeat):
        step = 0
        RandArr = np.random.permutation(training_iters)
        while step * batch_size < training_iters:
            global_step = step*batch_size
            batch_x, batch_y = next_batch(step*batch_size,batch_size)
            feed_dict = {img_input: batch_x, img_ans: batch_y}
            img_compressed = sess.run(img_com, feed_dict=feed_dict)
            batch_jpeg = get_jpeg_from_batch(img_compressed, h, w, qf)
            feed_dict = {img_input: batch_x, img_ans: batch_y, img_input_gen: batch_jpeg}
            
            img_fin = sess.run(img_final, feed_dict=feed_dict)
            img_inc, img_val = cor_compress(img_fin, batch_x)
            img_cor_d = cor_decompress(img_inc, img_val) 
            feed_dict = {img_input: batch_x, img_ans: batch_y, img_input_gen: batch_jpeg, img_input_cor: img_cor_d}
            
            for _ in range(cor_train):
            #if turn == 'gen':
                #pass
                #print(img_compressed.shape, img_compressed.dtype)
                sess.run(cor_op, feed_dict=feed_dict) 
            
            if step % display_step == 0:
                print("cor_lr: "+"{:.6f}".format(sess.run(r_opt._lr)))
                gloss = sess.run(gen_loss, feed_dict=feed_dict)
                rloss = sess.run(cor_loss, feed_dict=feed_dict)
                print("Step " + str(i) + ", Iter " + str(step*batch_size) + ", Minibatch GLoss= {:.6f}".format(gloss) + ", Minibatch RLoss= {:.6f}".format(rloss))
                raw_loss = get_mse(batch_x, get_jpeg_from_batch(batch_x, h, w, qf))
                valid_x, valid_y = valid_batch(0, n_valid)
                feed_dict={img_input:valid_x, img_ans:valid_y}
                #print(valid_x.shape)
                img_compressed = sess.run(img_com, feed_dict=feed_dict)
                #print(np.max(img_compressed[0]))
                batch_jpeg = get_jpeg_from_batch(img_compressed, h_val, w_val, qf)
                feed_dict = {img_input: valid_x, img_ans: valid_y, img_input_gen: batch_jpeg}
                 
                img_fin = sess.run(img_final, feed_dict=feed_dict)
                img_inc, img_val = cor_compress(img_fin, valid_x)
                img_cor_d = cor_decompress(img_inc, img_val) 
                feed_dict = {img_input: valid_x, img_ans: valid_y, img_input_gen: batch_jpeg, img_input_cor: img_cor_d}
            
                val_rloss = sess.run(cor_loss, feed_dict=feed_dict)
                val_gloss = sess.run(gen_loss, feed_dict=feed_dict)
                print("Testing GLoss : %.5f RLoss : %.5f" % (val_gloss, val_rloss))
                Log = open('Log_r.txt', 'a')
                Log.write("R: "+"{:.5f}".format(val_rloss)+"\n")
                Log.close()
                if min_loss>val_rloss:
                    min_loss = val_rloss
                    saver.save(sess, './model/r-model.ckpt-qf='+str(qf)+'-best')
            step+=1
        saver.save(sess, './model/r-model.ckpt-qf='+str(qf), global_step=i)
print('Minimum loss : ' + str(min_loss))
print('FINISHED!!!!!')
