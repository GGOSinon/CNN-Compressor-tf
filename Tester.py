import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
from ImageFunctions import *
from PIL import Image
Data = make_batch_with_image("../data/images/final/lena.bmp")
#Data = np.load("../data/data_val.npy")
test_size = Data.shape[0]
print(Data.shape)
n_prob = 20
h_prob, w_prob = 3, 3
#c, h, w = 1, 256, 256
c, h, w = Data.shape[3], Data.shape[1], Data.shape[2]
compress_rate = 4
qf = 5
#hh, ww = h/compress_rate, w/compress_rate

gpu_device = ['/device:GPU:0', '/device:GPU:1']
img_input = tf.placeholder(tf.float32, [None, h, w, c])
img_ans = tf.placeholder(tf.float32, [None, h, w, c]) 
img_input_cor = tf.placeholder(tf.float32, [None, h, w, c])
img_input_gen = tf.placeholder(tf.float32, [None, h, w, c])

def leaky_relu(x, alpha = 0.2):
    return tf.maximum(x, alpha * x)

def conv2d(x, W, b, stride = 1, act_func = 'ReLU', use_bn = True):
    strides = [1, stride, stride, 1]
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides, padding='SAME')
    x = tf.nn.bias_add(x, b)
    if act_func == 'LReLU': x = leaky_relu(x)
    if act_func == 'ReLU': x = tf.nn.relu(x)
    if act_func == 'TanH': x = tf.nn.tanh(x)
    if act_func == 'Sigmoid': x = tf.nn.sigmoid(x)
    if act_func == 'Softmax': x = tf.nn.softmax(x)
    if act_func == 'None': pass
    if use_bn: return slim.batch_norm(x, fused=False)
    else: return x

def com_net(x, weights, biases):
    x = tf.reshape(x, (-1, h, w, c))
    x = conv2d(x, weights['wc1'], biases['bc1'], act_func = 'LReLU', use_bn = False)
    for i in range(2, 3):
    	name_w = 'wc'+str(i)
    	name_b = 'bc'+str(i)
    	x = conv2d(x, weights[name_w], biases[name_b], act_func = 'LReLU')   
    res = conv2d(x, weights['wcx'], biases['bcx'], act_func='None', use_bn = False)
    return res

def gen_net(x, weights, biases):
    x = tf.reshape(x, (-1, h, w, c))
    x = conv2d(x, weights['wc1'], biases['bc1'], act_func = 'LReLU', use_bn = False)
    for i in range(1, 4/2):
    	x_input = x
        name_w = 'wc'+str(2*i)
        name_b = 'bc'+str(2*i)
        x = conv2d(x, weights[name_w], biases[name_b], act_func='LReLU')
        name_w = 'wc'+str(2*i+1)
        name_b = 'bc'+str(2*i+1)
        x = conv2d(x, weights[name_w], biases[name_b], act_func='None')
        x = leaky_relu(x_input + x)
    res = conv2d(x, weights['wcx'], biases['bcx'], act_func='None', use_bn = False)
    return res

def img_net(x, weights, biases):
    x = tf.reshape(x, (-1, h, w, c))
    x = conv2d(x, weights['wc1'], biases['bc1'], use_bn = False)
    for i in range(2, 6):
    	name_w = 'wc'+str(i)
    	name_b = 'bc'+str(i)
    	x = conv2d(x, weights[name_w], biases[name_b])
 
    res = conv2d(x, weights['wcx'], biases['bcx'], act_func='Softmax', use_bn = False)
    #fc1 = conv2d(conv3, weights['wd1'], biases['bd1'])
    #out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return res

def cor_net(x, weights, biases):
    x = tf.reshape(x, (-1, h, w, c))
    x = conv2d(x, weights['wc1'], biases['bc1'], use_bn = False)
    for i in range(2, 7):
    	name_w = 'wc'+str(i)
    	name_b = 'bc'+str(i)
    	x = conv2d(x, weights[name_w], biases[name_b]) 
    res = conv2d(x, weights['wcx'], biases['bcx'], act_func='None', use_bn = False)
    #fc1 = conv2d(conv3, weights['wd1'], biases['bd1'])
    #out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return res

def make_grad(s, e, name):
     return tf.get_variable(name, [3, 3, s, e], initializer=tf.contrib.layers.xavier_initializer())
    
def make_bias(x, name):
    return tf.get_variable(name, [x], initializer=tf.contrib.layers.xavier_initializer())

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

var_com = make_dict(5, 64, 'c', 1, 1)
print(var_com)
var_gen = make_dict(18, 64, 'g', 1, 1)
var_img = make_dict(18, 64, 'i', 1, 1)
var_cor = make_dict(18, 64, 'r', 1, 1)

#learning_rate = start_learning_rate

def next_batch(x, size=-1):
    data = []
    if size==-1:
        data.append(Data[x])
    else:
    	for i in range(size):
    		data.append(Data[x+i])
    data = np.array(data)
    data = data.reshape(-1, h, w, c)
    return data, data

# Define graph

# Com
img_com = com_net(img_input, var_com['weights'], var_com['biases'])
img_res = gen_net(img_com, var_gen['weights'], var_gen['biases'])

# Gen
#img_dscale = tf.image.resize_images(img_com, [hh, ww])
#img_dscale = tf.image.resize_images(img_input, [hh, ww])
#img_uscale = tf.placeholder([None, h, w, c])
#img_uscale = tf.image.resize_images(img_dscale, [h, w])

#img_res_gen = gen_net(img_uscale, var_gen['weights'], var_gen['biases'])
img_res_gen = gen_net(img_input_gen, var_gen['weights'], var_gen['biases'])
#img_final = img_res_gen + img_uscale
img_final = img_res_gen + img_input_gen
#img_final = img_uscale

# Img
img_prob = img_net(img_input, var_img['weights'], var_img['biases'])
img_delta = tf.reduce_mean(img_ans - img_final,axis=3,keep_dims=True)
img_ans_prob = tf.nn.l2_normalize(img_delta, dim=None)

# Cor
def cor_compress(img, p_img):
    sz_prob = img.shape[0]
    img_inc = np.zeros([sz_prob, n_prob], dtype=np.int32)
    img_val = np.zeros([sz_prob, n_prob], dtype=np.float32)

    for i in range(sz_prob):
        p_img_flatten = np.reshape(p_img[i], [-1])
        img_flatten = np.reshape(img[i], [-1])
        incides = np.argpartition(p_img_flatten,-n_prob)[-n_prob:]
        for j in range(n_prob):
            inc = incides[j]
            img_inc[i][j]=inc
            img_val[i][j]=img_flatten[inc]
    #img_res = np.array(img_res)
    #img_input_cor = tf.convert_to_tensor(img_input_cor_np)
    return img_inc, img_val

def cor_decompress(img_inc, img_val, img_fin):
    sz_img = img_inc.shape[0]
    img_decom = np.zeros([sz_img, h*w*c], dtype=np.float32)
    img_fin_flat = np.reshape(img_fin, [-1, h*w*c])
    for i in range(sz_img):
        for j in range(n_prob):
            inc,val = img_inc[i][j],img_val[i][j]-img_fin_flat[i][j] 
            img_decom[i][inc] = val
    img_decom = np.reshape(img_decom, [-1, h, w, c])
    return img_decom

def get_mse(x, y):
    res = np.reshape(x, [-1])
    ans = np.reshape(y, [-1])
    delta = res-ans
    mse = (delta**2).mean(axis=None)
    return mse

def get_psnr(x, y):
    mse = get_mse(x, y)
    return -10*(math.log(mse)/math.log(10))

img_cor = cor_net(img_input_cor,var_cor['weights'],var_cor['biases'])

img_real_final = img_final + img_cor
# Define loss and optimizer
com_loss = tf.losses.mean_squared_error(img_ans - img_com, img_res)
gen_loss = tf.losses.mean_squared_error(img_ans, img_final)
img_loss = tf.losses.mean_squared_error(img_ans_prob, img_prob)
#img_loss = tf.losses.softmax_cross_entropy(img_ans_prob, img_prob)
cor_loss = tf.losses.mean_squared_error(img_ans - img_final, img_cor)
# Initializing the variables
init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
max_acc = 0.

p_acc = []
#Log = open('Log_cg.txt', 'w')
#Log.close()

#saver = tf.train.Saver(var_list)
#saver = tf.train.Saver()

with sess: 
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, "./model/best/cg-model.ckpt-49")
    step = 0
    tot_mse = 0.
    while step < 1:#test_size:
        batch_x, batch_y = next_batch(step, test_size) 
        #h, w = batch_x.shape[1], batch_x.shape[2]
        #hh, ww = h//4, w//4
        h, w, H, W = 40, 40, 512//40, 512//40
        feed_dict = {img_input: batch_x, img_ans: batch_y}
        img_compressed = sess.run(img_input, feed_dict = feed_dict)
        #print(h, w)
        batch_jpeg = get_jpeg_from_batch(img_compressed, h, w, qf)
        print(batch_jpeg.shape)
        feed_dict = {img_input: batch_x, img_ans: batch_y, img_input_gen: batch_jpeg}
        img_fin = sess.run(img_final, feed_dict = feed_dict)
        #img_cor_prob, img_fin = sess.run([img_prob, img_final], feed_dict=feed_dict)
        #img_inc, img_val = cor_compress(batch_x, img_cor_prob)
        #img_cor_decom = cor_decompress(img_inc, img_val, img_fin)
        #feed_dict_cor = {img_input: batch_x, img_ans: batch_y, img_input_cor: img_cor_decom}
        #img_real_fin = sess.run(img_real_final, feed_dict=feed_dict_cor)
        Img_fin = np.reshape(img_fin, (H, W, h, w))
        Img_fin = merge_image(Img_fin)
        Img_fin = denormalize_img(Img_fin)
        Img_fin = np.reshape(Img_fin, (h*H, w*W))
        Img_fin = Image.fromarray(Img_fin, 'L')
        Img_fin.save('./final_result-'+str(step)+'.png')
        #Img_real_fin = Image.fromarray(img_real_fin[0], 'RGB')
        #Img_real_fin.save('./result/'+str(step)+'-cor.png')
        
        ans = batch_y
        closs, gloss = sess.run((com_loss, gen_loss), feed_dict=feed_dict)
        #print(img_fin, ans)
        for i in range(test_size):
        	tot_mse+=get_mse(img_fin[i], ans[i])
        print(gloss, get_mse(img_fin, ans))
        print(str(step)+": "+"{:.5f}".format(get_psnr(img_fin, ans)))
        step += test_size
    print(tot_mse/test_size)
print('FINISHED!!!!!')
