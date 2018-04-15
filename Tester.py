import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from math import log
from PIL import Image
Data = np.load("../data/data_val.npy")
#n_input = Data.shape[1]
test_size = Data.shape[0]

n_prob = 200
h_prob, w_prob = 3, 3
n_repeat = 15
c, h, w = 3, 256, 256
hh, ww = 64, 64

gpu_device = ['/device:GPU:0', '/device:GPU:1']
img_input = tf.placeholder(tf.float32, [None, h, w, c])
img_ans = tf.placeholder(tf.float32, [None, h, w, c]) 
img_input_cor = tf.placeholder(tf.float32, [None, h, w, c])

def conv2d(x, W, b, stride = 1, act_func = 'ReLU', use_bn = True):
    strides = [1, stride, stride, 1]
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides, padding='SAME')
    x = tf.nn.bias_add(x, b)
    if act_func == 'ReLU': x = tf.nn.relu(x)
    if act_func == 'TanH': x = tf.nn.tanh(x)
    if act_func == 'Sigmoid': x = tf.nn.sigmoid(x)
    if act_func == 'Softmax': x = tf.nn.softmax(x)
    if act_func == 'None': pass
    if use_bn: return slim.batch_norm(x)
    else: return x

def com_net(x, weights, biases):
    x = tf.reshape(x, (-1, h, w, c))
    #return x
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], use_bn = False)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    #conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    #conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    res = conv2d(conv2, weights['wcx'], biases['bcx'], act_func='None', use_bn = False)
    #fc1 = conv2d(conv3, weights['wd1'], biases['bd1'])
    #out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return res

def gen_net(x, weights, biases):
    x = tf.reshape(x, (-1, h, w, c))
    #print(x)
    #return x
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], use_bn = False)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    conv6 = conv2d(conv5, weights['wc6'], biases['bc6'])
    res = conv2d(conv6, weights['wcx'], biases['bcx'], act_func='None', use_bn = False)
    #fc1 = conv2d(conv3, weights['wd1'], biases['bd1'])
    #out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    #print("RES", res.shape)
    return res

def img_net(x, weights, biases):
    x = tf.reshape(x, (-1, h, w, c))
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], use_bn = False)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    #conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    #conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    res = conv2d(conv2, weights['wcx'], biases['bcx'], act_func='Softmax', use_bn = False)
    #fc1 = conv2d(conv3, weights['wd1'], biases['bd1'])
    #out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return res

def cor_net(x, weights, biases):
    x = tf.reshape(x, (-1, h, w, c))
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], use_bn = False)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4']) 
    conv5 = conv2d(conv4, weights['wc4'], biases['bc4'])
    conv6 = conv2d(conv5, weights['wc4'], biases['bc4'])
    res = conv2d(conv6, weights['wcx'], biases['bcx'], act_func='None', use_bn = False)
    #fc1 = conv2d(conv3, weights['wd1'], biases['bd1'])
    #out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return res

var_com={
'weights': {
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32]),name="w1c"),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 32]),name="w2c"),
    #'wc3': tf.Variable(tf.random_normal([5, 5, 32, 32]),name="w3c"),
    #'wc4': tf.Variable(tf.random_normal([5, 5, 32, 32]),name="w4c"),
    'wcx': tf.Variable(tf.random_normal([5, 5, 32, 3]),name="wxc")
},

'biases': {
    'bc1': tf.Variable(tf.random_normal([32]), name="b1c"),
    'bc2': tf.Variable(tf.random_normal([32]), name="b2c"),
    #'bc3': tf.Variable(tf.random_normal([32]), name="b3c"),
    #'bc4': tf.Variable(tf.random_normal([32]), name="b4c"),
    'bcx': tf.Variable(tf.random_normal([3]), name="bxc")
}
}

var_gen={
'weights' : {
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32]),name="w1g"),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 32]),name="w2g"),
    'wc3': tf.Variable(tf.random_normal([5, 5, 32, 32]),name="w3g"),
    'wc4': tf.Variable(tf.random_normal([5, 5, 32, 32]),name="w4g"),
    'wc5': tf.Variable(tf.random_normal([5, 5, 32, 32]),name="w5g"),
    'wc6': tf.Variable(tf.random_normal([5, 5, 32, 32]),name="w6g"),
    'wcx': tf.Variable(tf.random_normal([5, 5, 32, 3]),name="wxg")
},

'biases' : {
    'bc1': tf.Variable(tf.random_normal([32]), name="b1g"),
    'bc2': tf.Variable(tf.random_normal([32]), name="b2g"),
    'bc3': tf.Variable(tf.random_normal([32]), name="b3g"),
    'bc4': tf.Variable(tf.random_normal([32]), name="b4g"),
    'bc5': tf.Variable(tf.random_normal([32]), name="b5g"),
    'bc6': tf.Variable(tf.random_normal([32]), name="b6g"),
    'bcx': tf.Variable(tf.random_normal([3]), name="bxg")
}
}

var_img={
'weights' : {
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32]),name="w1i"),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 32]),name="w2i"),
    #'wc3': tf.Variable(tf.random_normal([5, 5, 32, 32]),name="w3i"),
    #'wc4': tf.Variable(tf.random_normal([5, 5, 32, 32]),name="w4i"),
    'wcx': tf.Variable(tf.random_normal([5, 5, 32, 1]),name="wxi")
},

'biases' : {
    'bc1': tf.Variable(tf.random_normal([32]), name="b1i"),
    'bc2': tf.Variable(tf.random_normal([32]), name="b2i"),
    #'bc3': tf.Variable(tf.random_normal([32]), name="b3i"),
    #'bc4': tf.Variable(tf.random_normal([32]), name="b4i"),
    'bcx': tf.Variable(tf.random_normal([1]), name="bxi")
}
}

var_cor={
'weights' : {
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32]),name="w1r"),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 32]),name="w2r"),
    'wc3': tf.Variable(tf.random_normal([5, 5, 32, 32]),name="w3r"),
    'wc4': tf.Variable(tf.random_normal([5, 5, 32, 32]),name="w4r"),
    'wc5': tf.Variable(tf.random_normal([5, 5, 32, 32]),name="w5r"),
    'wc6': tf.Variable(tf.random_normal([5, 5, 32, 32]),name="w6r"),
    'wcx': tf.Variable(tf.random_normal([5, 5, 32, 3]),name="wxr")
},

'biases' : {
    'bc1': tf.Variable(tf.random_normal([32]), name="b1r"),
    'bc2': tf.Variable(tf.random_normal([32]), name="b2r"),
    'bc3': tf.Variable(tf.random_normal([32]), name="b3r"),
    'bc4': tf.Variable(tf.random_normal([32]), name="b4r"),
    'bc5': tf.Variable(tf.random_normal([32]), name="b5r"),
    'bc6': tf.Variable(tf.random_normal([32]), name="b6r"),
    'bcx': tf.Variable(tf.random_normal([3]), name="bxr")
}
}

print(Data.shape)

def next_batch(pos):
    ans = []
    data = []
    data.append(Data[pos])
    data = np.array(data)
    data = data.reshape(-1, h, w, c)
    return data, data

# Define graph

# Com
img_com = com_net(img_input, var_com['weights'], var_com['biases'])
img_res = gen_net(img_com, var_gen['weights'], var_gen['biases'])

# Gen
img_dscale = tf.image.resize_images(img_com, [hh, ww])
img_uscale = tf.image.resize_images(img_dscale, [h, w])
img_res_gen = gen_net(img_uscale, var_gen['weights'], var_gen['biases'])
img_final = img_res_gen + img_uscale

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
    img_decom = np.zeros([sz_img, h*w*3], dtype=np.float32)
    img_fin_flat = np.reshape(img_fin, [-1, h*w*3])
    for i in range(sz_img):
        for j in range(n_prob):
            inc,val = img_inc[i][j],img_val[i][j]-img_fin_flat[i][j] 
            img_decom[i][inc] = val
    img_decom = np.reshape(img_decom, [-1, h, w, 3])
    return img_decom

img_cor = cor_net(img_input_cor,var_cor['weights'],var_cor['biases'])

img_real_final = img_final + img_cor
# Define loss and optimizer
com_loss = tf.losses.mean_squared_error(img_ans - img_com, img_res)
gen_loss = tf.losses.mean_squared_error(img_ans - img_uscale, img_res_gen)
img_loss = tf.losses.mean_squared_error(img_ans_prob, img_prob)
#img_loss = tf.losses.softmax_cross_entropy(img_ans_prob, img_prob)
cor_loss = tf.losses.mean_squared_error(img_ans - img_final, img_cor)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

Log = open('Log.txt', 'w')
Log.close()
saver = tf.train.Saver()

def get_psnr(x, y):
    res = np.reshape(x, [-1])
    ans = np.reshape(y, [-1])
    delta = res-ans
    #print(delta)
    mse = (delta**2).mean(axis=None)
    #mx = np.amax(ans)
    #print(mse)
    return - 10*(log(mse)/log(10))

with sess:
    saver.restore(sess, "./model/CNN-model.ckpt-6")
    step = 0
    while step < test_size:
        batch_x, batch_y = next_batch(step)
        #print(batch_x)
        feed_dict = {img_input: batch_x, img_ans: batch_y}
        img_cor_prob, img_fin = sess.run([img_prob, img_final], feed_dict=feed_dict)
        img_inc, img_val = cor_compress(batch_x, img_cor_prob)
        img_cor_decom = cor_decompress(img_inc, img_val, img_fin)
        feed_dict_cor = {img_input: batch_x, img_ans: batch_y, img_input_cor: img_cor_decom}
        img_real_fin = sess.run(img_real_final, feed_dict=feed_dict_cor)
        Img_fin = Image.fromarray(img_fin[0], 'RGB')
        Img_fin.save('./result/'+str(step)+'.png')
        Img_real_fin = Image.fromarray(img_real_fin[0], 'RGB')
        Img_real_fin.save('./result/'+str(step)+'-cor.png')

        ans = batch_y
        print(str(step)+": "+"{:.5f}".format(get_psnr(img_fin, ans)))
        #print(str(step)+": "+"{:.5f}".format(get_psnr(img_real_fin, ans)))
        step += 1

print('FINISHED!!!!!')
