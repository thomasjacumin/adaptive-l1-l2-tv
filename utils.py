import numpy as np
from scipy import signal
import operators
from PIL import Image
import math

def PSNR(u, v, w, h):
  EQM = np.sum(np.power(u-v,2))/w/h
  return 10*np.log10(1/EQM)

def UnifMeanFilter(mesh, f, ww, w, h):
  kern = np.ones([ww,ww])/ww**2
  return signal.convolve2d(f.reshape([h,w]), kern, boundary='symm', mode='same').flatten()

def UnifGaussianFilter(mesh, f, gamma, w, h):
  def heatKernel(mesh, x0, y0, t):
    kernel = np.zeros(w*h)
    for e in mesh.elements:
      kernel[e.indice] = 1/(4*np.pi*t)*np.exp(-((x0-e.x)**2 + (y0-e.y)**2)/(4*t))
    return kernel/mesh.dofs
  
  kern = heatKernel(mesh, 0.5, 0.5, gamma)
  return signal.convolve2d(f.reshape([h,w]), kern.reshape([h,w]), boundary='symm', mode='same').flatten()

def TV(mesh, v, alpha):
    gradOp = operators.gradOperator(mesh)
    gradV = gradOp@v
    res = 0
    for i in range(0,mesh.dofs):
        res = res + mesh.elements[i].dx**2*alpha[i]*np.sqrt(gradV[i]**2 + gradV[i+mesh.dofs]**2)
    
    return res

def I(mesh, v, f, alpha):
    L2 = 0
    for i in range(0,mesh.dofs):
        L2 = L2 + mesh.elements[i].dx**2*(v[i]-f[i])**2
    
    return TV(mesh, v, alpha) + L2

def D(mesh, q, f):
    divOp = operators.divOperator(mesh)
    divQ = divOp@q
    ret = 0
    for i in range(0,mesh.dofs):
        ret = ret - mesh.elements[i].dx**2*(divQ[i] + f[i])**2 + mesh.elements[i].dx**2*f[i]**2
    return ret

def isotropicRatio(mesh, f):
    ratio = np.zeros(mesh.dofs)
    for e in mesh.elements: 
        if len(e.eastElements) > 0 and len(e.southElements) > 0:
            if ( np.sqrt( (f[e.eastElements[0].indice] - f[e.indice])**2 + (f[e.southElements[0].indice] - f[e.indice])**2 ) + np.sqrt( (f[e.eastElements[0].indice] - f[e.indice])**2 ) + np.sqrt( (f[e.southElements[0].indice] - f[e.indice])**2 ) ) > 0:
                ratio[e.indice] = 2*np.sqrt( (f[e.eastElements[0].indice] - f[e.indice])**2 + (f[e.southElements[0].indice] - f[e.indice])**2 )/( np.sqrt( (f[e.eastElements[0].indice] - f[e.indice])**2 + (f[e.southElements[0].indice] - f[e.indice])**2 ) + np.sqrt( (f[e.eastElements[0].indice] - f[e.indice])**2 ) + np.sqrt( (f[e.southElements[0].indice] - f[e.indice])**2 ) )
    return ratio


def L2Norm(mesh, v):
    l2 = 0
    for e in mesh.elements:
        l2 = l2 + e.dx*e.dy*v[e.indice]**2
    return np.sqrt(l2)

import colorsys
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def vectorToColor(w, rhoMax):
    rho = np.sqrt(w[0]**2 + w[1]**2) / rhoMax
    theta = np.arctan2(w[1], w[0])
    if theta < 0:
        theta = theta + 2*math.pi
    return colorsys.hsv_to_rgb(theta / (2*math.pi), np.clip(rho, 0, 1), 1)

def opticalFlowToRGB(mesh, u, v, rhoMax=None):
    if rhoMax == None:
        rhoMax = max(np.max(np.power(u,2) + np.power(v,2)), 1)
    oF = np.zeros([mesh.dofs,3])
    for e in mesh.elements:
        color = vectorToColor([u[e.indice],v[e.indice]], np.sqrt(rhoMax))
        oF[e.indice,:] = color[:]
    return oF

def showQMeshOF(mesh, oF, pathname=None, title=None, displayEdges=True):
    elements = mesh.elements
    X = [e.x for e in mesh.elements]
    Y = [e.y for e in mesh.elements]
    aspect = np.max(X)/np.max(Y)
    fig, ax = plt.subplots(figsize=(10*aspect,10))
    
    padding = 1
    ax.set_ylim( np.max(Y) + padding, np.min(Y) - padding)
    ax.set_xlim( np.min(X) - padding, np.max(X) + padding)

    if title:
        ax.text(0, -0.01, title)

    for e in elements:
        dx = e.dx
        dy = e.dy
        
        if displayEdges:
            ax.add_patch(Rectangle((e.x-dx/2, e.y-dy/2), dx, dy, edgecolor = 'blue', facecolor = (oF[e.indice,0],oF[e.indice,1],oF[e.indice,2]), fill=True))
        else:
            ax.add_patch(Rectangle((e.x-dx/2, e.y-dy/2), dx, dy, facecolor = (oF[e.indice,0],oF[e.indice,1],oF[e.indice,2]), fill=True)) 

    plt.tight_layout()
    if pathname != None:
        plt.savefig(pathname)
    plt.show()

def eval(mesh, f, x, y):
    for e in mesh.elements:
        if x < e.x + e.dx/2 and x > e.x - e.dx/2 and y < e.y + e.dy/2 and y > e.y - e.dy/2:
            return f[e.indice]
    return 0

def apply(f1, u, v, w, h):
    x = np.zeros([w*h])
    for i in range(0,h):
      for j in range(0,w):
        tildI = i + v[i*w+j]
        tildJ = j + u[i*w+j]
        dI = tildI-int(tildI)
        dJ = tildJ-int(tildJ)

        w1 = (1-dI)*(1-dJ)
        w2 = dJ*(1-dI)
        w3 = dI*dJ
        w4 = (1-dJ)*dI

        if tildI >= h:
          tildI = h-1
        if tildJ >= w:
          tileJ = w-1
        if tildI < 0:
          tildI = 0
        if tildJ < 0:
          tildJ = 0

        if int(tildI)*w+int(tildJ)<w*h and i*w+j<w*h:
          if int(tildI) < h-1 and int(tildJ) < w-1: # not on the left or bottom boundaries
            x[i*w+j] = w1*f1[int(tildI)*w + int(tildJ)]
            x[i*w+j] = x[i*w+j] + w2*f1[int(tildI)*w + int(tildJ)+1]
            x[i*w+j] = x[i*w+j] + w3*f1[int(tildI+1)*w + int(tildJ)+1]
            x[i*w+j] = x[i*w+j] + w4*f1[int(tildI+1)*w + int(tildJ)]
          elif int(tildI) < h-1 and int(tildJ) == w-1: # left boundary
            x[i*w+j] = w1*f1[int(tildI)*w + int(tildJ)]
            x[i*w+j] = x[i*w+j] + w2*f1[int(tildI)*w + int(tildJ)]
            x[i*w+j] = x[i*w+j] + w3*f1[int(tildI+1)*w + int(tildJ)]
            x[i*w+j] = x[i*w+j] + w4*f1[int(tildI+1)*w + int(tildJ)]
          elif int(tildI) == h-1 and int(tildJ) < w-1: # bottom boundary
            x[i*w+j] = w1*f1[int(tildI)*w + int(tildJ)]
            x[i*w+j] = x[i*w+j] + w2*f1[int(tildI)*w + int(tildJ)+1]
            x[i*w+j] = x[i*w+j] + w3*f1[int(tildI)*w + int(tildJ)+1]
            x[i*w+j] = x[i*w+j] + w4*f1[int(tildI)*w + int(tildJ)]
          else: # bottom-left corner
            x[i*w+j] = w1*f1[int(tildI)*w + int(tildJ)]
            x[i*w+j] = x[i*w+j] + w2*f1[int(tildI)*w + int(tildJ)]
            x[i*w+j] = x[i*w+j] + w3*f1[int(tildI)*w + int(tildJ)]
            x[i*w+j] = x[i*w+j] + w4*f1[int(tildI)*w + int(tildJ)]

    return x

def openFlo(pathname):
    f = open(pathname, 'rb')
    magic = np.fromfile(f, np.float32, count=1)[0]
    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    w = np.fromfile(f, np.int32, count=1)[0]
    h = np.fromfile(f, np.int32, count=1)[0]
    data = np.fromfile(f, np.float32)
    data_2D = np.reshape(data, newshape=(h,w,2));
    x = data_2D[...,0].flatten()
    y = data_2D[...,1].flatten()
    return w, h, x, y

def saveFlo(w,h,u,v,pathname):
    f = open(pathname, 'wb')
    np.array([202021.25], dtype=np.float32).tofile(f)
    np.array([w,h], dtype=np.int32).tofile(f)
    data = np.zeros([w*h,2])
    data[:,0] = u
    data[:,1] = v
    # data.reshape([h,w,2])
    np.array(data, dtype=np.float32).tofile(f)

def saveImage(w,h,img,pathname):
    Image.fromarray(np.uint8(255*img.reshape([h,w])), 'L').save(pathname)

def EE(w, h, u, v, uGT, vGT):
    _EE  = np.sqrt( (u-uGT)**2 + (v-vGT)**2 )
    _EE_ignore = []
    for i in range(0, w*h):
        if _EE[i] <= 50:
            _EE_ignore.append(_EE[i])
    AEE  = np.sum( _EE_ignore )/len(_EE_ignore)
    SDEE = np.sqrt( np.sum((_EE_ignore - AEE)**2 )/len(_EE_ignore) )
    return AEE, SDEE

def AE(w, h, u, v, uGT, vGT):
    _AE  = np.arccos( (1.0 + u*uGT + v*vGT)/(np.sqrt(1.0 + u**2 + v**2)*np.sqrt(1.0 + uGT**2 + vGT**2)) )
    _AE_ignore = []
    for i in range(0, w*h):
        if math.isnan(_AE[i]) == False:
            _AE_ignore.append(_AE[i])
    AAE  = np.sum( _AE_ignore )/len(_AE_ignore)
    SDAE = np.sqrt( np.sum((_AE_ignore - AAE)**2 )/len(_AE_ignore) )
    return AAE, SDAE