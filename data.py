import numpy as np
from PIL import Image

def horizontal(w,h):
    img = 0*np.ones([w,h])
    for i in range(0,int(h/2)):
    # for i in range(0,h):  
        for j in range(0,w):
        # if (i-h/2)**2 + (j-w/2)**2 <= r**2:
            img[i,j] = 1
    return img.flatten()

def square(w,h):
    img = 0.1*np.ones([w,h])
    for i in range(int(h/4),int(3*h/4)):
        for j in range(int(w/4),int(3*w/4)):
            img[i,j] = 1.
    return img.flatten()

def circle(w,h,r):
    img = 0*np.ones([w,h])
    # for i in range(0,int(h/2)):
    for i in range(0,h):  
        for j in range(0,w): 
            if (i-h/2)**2 + (j-w/2)**2 <= r**2:
                img[i,j] = 1
    return img.flatten()

def image(pathname):
    f = np.asarray(Image.open(pathname).convert('L'))
    w = np.size(f,1)
    h = np.size(f,0)
    f = f / 255
    return f.flatten(), w, h