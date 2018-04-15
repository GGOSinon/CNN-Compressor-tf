from PIL import Image, ImageOps
import fnmatch
import os
import copy
import numpy as np

size0 = 180, 180
size = 40, 40
skip_pixel = 20, 20

def image_to_np(img):
    image_np = np.fromstring(img.tobytes(), dtype=np.uint8).astype(float)
    image_np = image_np.reshape((size[0], size[1], 1))
    image_np/=255.
    return image_np

def make_data_with_image(image_path, isTest = False):
    #print(image_path)
    image = Image.open(image_path)
    new_image_list = []
    h = image.size[0]
    w = image.size[1]
    #print(h, w)
    #print(len(image.tobytes()), h*w)
    if len(image.tobytes()) == h*w: is_gray = True
    else: is_gray = False
    if is_gray==False: image = ImageOps.grayscale(image)
    '''
    if h < size[0]:
        image = image.resize((size[0], w), Image.ANTIALIAS)
        h = size[0]
    if w < size[1]:
        image = image.resize((h, size[1]), Image.ANTIALIAS)
        w = size[1]
    '''
    #if isTest:
    #	if h<w: image = image.crop((0, (w-h)//2, h, (w+h)//2))
    #	if h>w: image = image.crop(((h-w)//2, 0, (h+w)//2, w))
    #    return [image_to_np(image)]
    image = image.crop(((h-size0[0])//2, (w-size0[1])//2, (h+size0[0])//2, (w+size0[1])//2))
    
    h, w = size0
    #print(h, w)
    #print(image.size)
    H = (h-size[0])//skip_pixel[0]
    W = (w-size[1])//skip_pixel[1]
    for i in range(H+1):
        sh = i*skip_pixel[0]
        for j in range(W+1):
            sw = j*skip_pixel[1]
            cropped_image = image.crop((sh, sw, sh+size[0], sw+size[1]))
            for k in range(0, 360, 90):
            	#print(new_image.size)
            	new_image = cropped_image
                new_image_np = image_to_np(new_image.rotate(k))
                new_image_list.append(new_image_np)
                #print(new_image_np.shape)
                new_image = ImageOps.mirror(cropped_image)
                new_image_np = image_to_np(new_image.rotate(k))
                new_image_list.append(new_image_np)
                #print(new_image_np.shape)
            #print(new_image_np.shape)
    new_image_list = np.array(new_image_list).astype(np.float32)
    #print(new_image_list.shape)
    return new_image_list

matches = []
cnt = 0
for root, dirnames, filenames in os.walk('./images'):
    for filename in fnmatch.filter(filenames, '*.jpg'):
        matches.append(os.path.join(root, filename)) 
        cnt+=1

print("%d files found" % cnt)

tot_image = []
val_image = []

cnt0=32
for i in range(cnt0):
    F = matches[i]
    if i%100==0: print("Step "+str(i)+" completed")
    images = make_data_with_image(F)
    val_image.extend([images[0]])

cnt=400
for i in range(cnt0, cnt0+cnt):
    F = matches[i]
    if i%100==0: print("Step "+str(i)+" completed")
    images = make_data_with_image(F)
    tot_image.extend(images)

#print(tot_image)
#for image in tot_image:
    #print(image.shape)
tot_image = np.array(tot_image)
print(tot_image.shape)
val_image = np.array(val_image)
print(val_image.shape)
np.save("data.npy", tot_image)
np.save("data_val.npy", val_image)

