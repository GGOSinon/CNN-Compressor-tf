import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math

Data = np.load("../data/data.npy")
Data_val = np.load("../data/data_val.npy")
n_input = Data.shape[1]
training_iters = Data.shape[0]
n_valid = Data_val.shape[0]

#training_iters = 2560
n_valid = 32
cg_learning_rate = 0.01
ir_learning_rate = 0.01
cg_learning_end = 0.0001
ir_learning_end = 0.0001
n_repeat = 50
n_decay = 10
cg_decay_rate = pow(cg_learning_end/cg_learning_rate,1./(n_repeat*n_decay))
ir_decay_rate = pow(ir_learning_end/ir_learning_rate,1./(n_repeat*n_decay))
#print(pow(cg_decay_rate, n_repeat))
batch_size = 32
display_step = 10
tot_step = training_iters/batch_size
decay_step = tot_step//10

n_prob = 20
h_prob, w_prob = 3, 3
c, h, w = 1, 40, 40
hh, ww = 10, 10

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
    x = conv2d(x, weights['wc1'], biases['bc1'], use_bn = False)
    for i in range(2, 3):
    	name_w = 'wc'+str(i)
    	name_b = 'bc'+str(i)
    	x = conv2d(x, weights[name_w], biases[name_b])   
    res = conv2d(x, weights['wcx'], biases['bcx'], act_func='None', use_bn = False)
    return res

def gen_net(x, weights, biases):
    x = tf.reshape(x, (-1, h, w, c))
    x = conv2d(x, weights['wc1'], biases['bc1'], use_bn = False)
    for i in range(2, 18):
    	name_w = 'wc'+str(i)
    	name_b = 'bc'+str(i)
    	x = conv2d(x, weights[name_w], biases[name_b])
    res = conv2d(x, weights['wcx'], biases['bcx'], act_func='None', use_bn = False)
    return res

def img_net(x, weights, biases):
    x = tf.reshape(x, (-1, h, w, c))
    x = conv2d(x, weights['wc1'], biases['bc1'], use_bn = False)
    for i in range(2, 3):
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
    for i in range(2, 6):
    	name_w = 'wc'+str(i)
    	name_b = 'bc'+str(i)
    	x = conv2d(x, weights[name_w], biases[name_b]) 
    res = conv2d(x, weights['wcx'], biases['bcx'], act_func='None', use_bn = False)
    #fc1 = conv2d(conv3, weights['wd1'], biases['bd1'])
    #out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return res

var_com={
'weights': {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32]),name="w1c"),
    'wc2': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w2c"),
    #'wc3': tf.Variable(tf.random_normal([5, 5, 32, 32]),name="w3c"),
    #'wc4': tf.Variable(tf.random_normal([5, 5, 32, 32]),name="w4c"),
    'wcx': tf.Variable(tf.random_normal([3, 3, 32, 1]),name="wxc")
},

'biases': {
    'bc1': tf.Variable(tf.random_normal([32]), name="b1c"),
    'bc2': tf.Variable(tf.random_normal([32]), name="b2c"),
    #'bc3': tf.Variable(tf.random_normal([32]), name="b3c"),
    #'bc4': tf.Variable(tf.random_normal([32]), name="b4c"),
    'bcx': tf.Variable(tf.random_normal([1]), name="bxc")
}
}

var_gen={
'weights' : {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32]),name="w1g"),
    'wc2': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w2g"),
    'wc3': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w3g"), 
    'wc4': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w4g"),
    'wc5': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w5g"),
    'wc6': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w6g"),
    'wc7': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w7g"),
    'wc8': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w8g"),
    'wc9': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w9g"),
    'wc10': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w10g"),
    'wc11': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w11g"),
    'wc12': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w12g"),
    'wc13': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w13g"),
    'wc14': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w14g"),
    'wc15': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w15g"),
    'wc16': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w16g"),
    'wc17': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w17g"),
    'wc18': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w18g"),
    'wcx': tf.Variable(tf.random_normal([3, 3, 32, 1]),name="wxg")
},

'biases' : {
    'bc1': tf.Variable(tf.random_normal([32]), name="b1g"),
    'bc2': tf.Variable(tf.random_normal([32]), name="b2g"),
    'bc3': tf.Variable(tf.random_normal([32]), name="b3g"),
    'bc4': tf.Variable(tf.random_normal([32]), name="b4g"),
    'bc5': tf.Variable(tf.random_normal([32]), name="b5g"),
    'bc6': tf.Variable(tf.random_normal([32]), name="b6g"),
    'bc7': tf.Variable(tf.random_normal([32]), name="b7g"),
    'bc8': tf.Variable(tf.random_normal([32]), name="b8g"),
    'bc9': tf.Variable(tf.random_normal([32]), name="b9g"),
    'bc10': tf.Variable(tf.random_normal([32]), name="b10g"),
    'bc11': tf.Variable(tf.random_normal([32]), name="b11g"),
    'bc12': tf.Variable(tf.random_normal([32]), name="b12g"),
    'bc13': tf.Variable(tf.random_normal([32]), name="b13g"),
    'bc14': tf.Variable(tf.random_normal([32]), name="b14g"),
    'bc15': tf.Variable(tf.random_normal([32]), name="b15g"),
    'bc16': tf.Variable(tf.random_normal([32]), name="b16g"),
    'bc17': tf.Variable(tf.random_normal([32]), name="b17g"),
    'bc18': tf.Variable(tf.random_normal([32]), name="b18g"),
    'bcx': tf.Variable(tf.random_normal([1]), name="bxg")
}
}

var_img={
'weights' : {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32]),name="w1i"),
    'wc2': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w2i"),
    'wc3': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w3i"), 
    'wc4': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w4i"),
    'wc5': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w5i"),
    'wc6': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w6i"),
    'wc7': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w7i"),
    'wc8': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w8i"),
    'wc9': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w9i"),
    'wc10': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w10i"),
    'wc11': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w11i"),
    'wc12': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w12i"),
    'wc13': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w13i"),
    'wc14': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w14i"),
    'wc15': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w15i"),
    'wc16': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w16i"),
    'wc17': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w17i"),
    'wc18': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w18i"),
    'wcx': tf.Variable(tf.random_normal([3, 3, 32, 1]),name="wxi")
},

'biases' : {
    'bc1': tf.Variable(tf.random_normal([32]), name="b1i"),
    'bc2': tf.Variable(tf.random_normal([32]), name="b2i"),
    'bc3': tf.Variable(tf.random_normal([32]), name="b3i"),
    'bc4': tf.Variable(tf.random_normal([32]), name="b4i"),
    'bc5': tf.Variable(tf.random_normal([32]), name="b5i"),
    'bc6': tf.Variable(tf.random_normal([32]), name="b6i"),
    'bc7': tf.Variable(tf.random_normal([32]), name="b7i"),
    'bc8': tf.Variable(tf.random_normal([32]), name="b8i"),
    'bc9': tf.Variable(tf.random_normal([32]), name="b9i"),
    'bc10': tf.Variable(tf.random_normal([32]), name="b10i"),
    'bc11': tf.Variable(tf.random_normal([32]), name="b11i"),
    'bc12': tf.Variable(tf.random_normal([32]), name="b12i"),
    'bc13': tf.Variable(tf.random_normal([32]), name="b13i"),
    'bc14': tf.Variable(tf.random_normal([32]), name="b14i"),
    'bc15': tf.Variable(tf.random_normal([32]), name="b15i"),
    'bc16': tf.Variable(tf.random_normal([32]), name="b16i"),
    'bc17': tf.Variable(tf.random_normal([32]), name="b17i"),
    'bc18': tf.Variable(tf.random_normal([32]), name="b18i"),
    'bcx': tf.Variable(tf.random_normal([1]), name="bxi")
}
}

var_cor={
'weights' : {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32]),name="w1r"),
    'wc2': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w2r"),
    'wc3': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w3r"), 
    'wc4': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w4r"),
    'wc5': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w5r"),
    'wc6': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w6r"),
    'wc7': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w7r"),
    'wc8': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w8r"),
    'wc9': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w9r"),
    'wc10': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w10r"),
    'wc11': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w11r"),
    'wc12': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w12r"),
    'wc13': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w13r"),
    'wc14': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w14r"),
    'wc15': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w15r"),
    'wc16': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w16r"),
    'wc17': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w17r"),
    'wc18': tf.Variable(tf.random_normal([3, 3, 32, 32]),name="w18r"),
    'wcx': tf.Variable(tf.random_normal([3, 3, 32, 1]),name="wxr")
},

'biases' : {
    'bc1': tf.Variable(tf.random_normal([32]), name="b1r"),
    'bc2': tf.Variable(tf.random_normal([32]), name="b2r"),
    'bc3': tf.Variable(tf.random_normal([32]), name="b3r"),
    'bc4': tf.Variable(tf.random_normal([32]), name="b4r"),
    'bc5': tf.Variable(tf.random_normal([32]), name="b5r"),
    'bc6': tf.Variable(tf.random_normal([32]), name="b6r"),
    'bc7': tf.Variable(tf.random_normal([32]), name="b7r"),
    'bc8': tf.Variable(tf.random_normal([32]), name="b8r"),
    'bc9': tf.Variable(tf.random_normal([32]), name="b9r"),
    'bc10': tf.Variable(tf.random_normal([32]), name="b10r"),
    'bc11': tf.Variable(tf.random_normal([32]), name="b11r"),
    'bc12': tf.Variable(tf.random_normal([32]), name="b12r"),
    'bc13': tf.Variable(tf.random_normal([32]), name="b13r"),
    'bc14': tf.Variable(tf.random_normal([32]), name="b14r"),
    'bc15': tf.Variable(tf.random_normal([32]), name="b15r"),
    'bc16': tf.Variable(tf.random_normal([32]), name="b16r"),
    'bc17': tf.Variable(tf.random_normal([32]), name="b17r"),
    'bc18': tf.Variable(tf.random_normal([32]), name="b18r"),
    'bcx': tf.Variable(tf.random_normal([1]), name="bxr")
}
}

print(Data.shape)
print(Data_val.shape)
global_step = 0
#learning_rate = start_learning_rate

print(training_iters)
print(n_valid)

def valid_batch(prev, size):
	ans = []
	data = []
	for i in range(size):
		if i+prev>=n_valid: continue
		data.append(Data_val[i+prev])
	data = np.array(data)
	data = data.reshape(-1, h, w, c)
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
    img_decom = np.zeros([sz_img, h*w*c], dtype=np.float32)
    img_fin_flat = np.reshape(img_fin, [-1, h*w*c])
    for i in range(sz_img):
        for j in range(n_prob):
            inc,val = img_inc[i][j],img_val[i][j]-img_fin_flat[i][j] 
            img_decom[i][inc] = val
    img_decom = np.reshape(img_decom, [-1, h, w, c])
    return img_decom

img_cor = cor_net(img_input_cor,var_cor['weights'],var_cor['biases'])

img_real_final = img_final + img_cor
# Define loss and optimizer
com_loss = tf.losses.mean_squared_error(img_ans - img_com, img_res)
gen_loss = tf.losses.mean_squared_error(img_ans - img_uscale, img_res_gen)
img_loss = tf.losses.mean_squared_error(img_ans_prob, img_prob)
#img_loss = tf.losses.softmax_cross_entropy(img_ans_prob, img_prob)
cor_loss = tf.losses.mean_squared_error(img_ans - img_final, img_cor)

cg_opt = tf.train.AdamOptimizer(learning_rate = cg_learning_rate)
ir_opt = tf.train.AdamOptimizer(learning_rate = ir_learning_rate)

com_op = cg_opt.minimize(com_loss, var_list = var_com)
gen_op = cg_opt.minimize(gen_loss, var_list = var_gen)
img_op = ir_opt.minimize(img_loss, var_list = var_img)
cor_op = ir_opt.minimize(cor_loss, var_list = var_cor)

# Initializing the variables
init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
max_acc = 0.

p_acc = []
Log = open('Log_cg.txt', 'w')
Log.close()
saver = tf.train.Saver(max_to_keep = None)

com_train = 1
gen_train = 2

img_train = 1
cor_train = 1

with sess:
    sess.run(init)
    # Keep training until reach max iterations
    for i in range(n_repeat):
        step = 1
        RandArr = np.random.permutation(training_iters)
        while step * batch_size < training_iters:
            global_step = step*batch_size
            batch_x, batch_y =next_batch(step*batch_size,batch_size)
            feed_dict = {img_input: batch_x, img_ans: batch_y}
            for _ in range(com_train):
            	#print(batch_x.shape, batch_x.dtype)
                sess.run(com_op, feed_dict=feed_dict)
            for _ in range(gen_train):
                #print(img_compressed.shape, img_compressed.dtype)
            	sess.run(gen_op, feed_dict=feed_dict) 
            
            if step % display_step == 0:
                closs = sess.run(com_loss, feed_dict=feed_dict)
                gloss = sess.run(gen_loss, feed_dict=feed_dict)
                #closs /= batch_size
                #gloss /= batch_size
                print ("Step " + str(i) + ", Iter " + str(step*batch_size) + ", Minibatch CLoss= " + "{:.6f}".format(closs) + ", Minibatch GLoss= " + "{:.6f}".format(gloss))
                tot_closs = 0.
                tot_gloss = 0.

                valid_x, valid_y = valid_batch(0, n_valid)
                feed_dict_val={img_input:valid_x,img_ans:valid_y}
                val_closs = sess.run(com_loss, feed_dict=feed_dict_val)
                val_gloss = sess.run(gen_loss, feed_dict=feed_dict_val)
                print ("Testing CLoss : %.5f, GLoss : %.5f" % (val_closs, val_gloss))
                Log = open('Log_cg.txt', 'a')
                Log.write("C: "+"{:.5f}".format(val_closs)+" G: "+"{:.5f}".format(val_gloss)+"\n")
                Log.close()
            if step % decay_step==0:cg_learning_rate*=cg_decay_rate
            step+=1
        saver.save(sess, './model/cg-model.ckpt', global_step=i)

print('FINISHED!!!!!')
