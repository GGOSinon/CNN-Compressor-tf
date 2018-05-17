from ImageFunctions import *

#Data = np.load("../data/data_val.npy")
h, w = 40, 40
H, W = 512//h, 512//w
skip_pixel = [40, 40]
Data = make_batch_with_path("../data/images/final/lena.bmp", skip_pixel = skip_pixel)
print(Data.shape)
#img_ans = Data
qf = 10

h, w = Data.shape[1], Data.shape[2]

img0 = Data
img0 = np.reshape(img0, (H, W, h, w))
'''
for i in range(H):
    for j in range(W):
        K+=1
        x = np.reshape(img0[i][j], (h, w))
        x = denormalize_img(x)
        x = Image.fromarray(x, 'L')
        x.save('prev-'+str(K)+'.png')
'''
img0 = merge_image(img0)
Data = img0

img_ans = img0
img0 = denormalize_img(img0)#(Data*255).astype(np.uint8)
img0 = np.reshape(img0, (H*h,W*w))
img0 = Image.fromarray(img0, 'L')
img0.save('prev.png')

img = get_jpeg_from_batch(Data, h, w, qf)
img = np.reshape(img, (H, W, h, w))
img = merge_image(img)
img_com = img
img = denormalize_img(img)#img = (img*255).astype(np.uint8)
img = np.reshape(img, (H*h,W*w))
img = Image.fromarray(img, 'L')
img.save('result.png')

print(get_mse(img_ans, img_com))
print(get_psnr(img_ans, img_com))
