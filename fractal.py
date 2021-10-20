import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300
from PIL import Image,ImageFilter
import numpy as np
import math


def dbc(im, s, mode="standard", debug=False):
    (width, height) = im.shape
    assert(width == height)
    M = width
    # grid size must be bigger than 2 and least than M/2    
    G = 256 # range for dtype=np.uint8 # better way?
    assert(s >= 2)
    assert(s <= M//2)
    ngrid = math.ceil(M / s)
    h = G*(s / M) # box height
    grid = np.zeros((ngrid,ngrid), dtype='int32')
    
    #iterate through larger grid
    for i in range(ngrid):
        for j in range(ngrid):
            maxg = 0
            ming = 255
            #iterate through each pixel in sub-grid 
            for k in range(i*s, min((i+1)*s, M)):
                for l in range(j*s, min((j+1)*s, M)):
                    if im[k, l] > maxg:
                        maxg = im[k, l]
                    if im[k, l] < ming:
                        ming = im[k, l]
            
            # box counting methods
            if mode == "standard":
                grid[i,j] = math.ceil(maxg/h) - math.ceil(ming/h) + 1
            if mode == "shifting":    
                grid[i,j] = math.ceil((maxg-ming+1)/h)
    
    if debug:
        plt.title(f"mode: {mode} \n scale: {s}")
        plt.imshow(grid)
        plt.show()
    
    Ns = grid.sum()
    
    return Ns


if __name__ == '__main__':
    path = str(input("Enter path to image:"))
    # make image if nothing passed
    if path == "":
        from skimage.morphology import disk
        image = disk(20)
        rng = np.random.default_rng(seed=0)
        random_grid = rng.random(image.shape)
        im = (image * random_grid * 255).astype(int)
        image = im 
        
    else:
        from pathlib import Path
        # path = Path(r"./sierpienski_triangle.jpg")
        path = Path(r"./waves.jpg")
        # path = Path(r"./clouds.jpg")
        # path = Path(r"./ocean.jpg")
        # path = Path(path)
        image = Image.open(path) # Brodatz/D1.gif
        image = image.convert('L')
        # invert, make into abox, and set as binary with uin8 range
        
        image = np.asarray(image)
        # image = np.invert(np.asarray(image, dtype=(np.uint8)))
        im_min = min(image.shape)
        image = image[:im_min, :im_min]
        # image = ((image > 100).astype(int) * 255).astype(np.uint8)
        
    
    # (imM, _) = image.size
    (imM, _) = image.shape
    
    # calculate Nr and r
    Nr = []
    r = []
    print("|\tNr\t|\tr\t|S\t|")
    a = 2
    b = imM//2
    nval = 20
    lnsp = np.linspace(1,math.log(b,a),nval)
    sval  = a**lnsp
	
    for S in sval:#range(2,imM//2,(imM//2-2)//100):
        Ns = dbc(image, int(S), debug=True)
        Nr.append(Ns)
        R = S/imM
        # r.append(S) # I think this should be R
        r.append(R)
        print("|%10d\t|%10f\t|%4d\t|"% (Ns,R,S))
	
	
    # calculate log(Nr) and log(1/r)    
    y = np.log(np.array(Nr))
    x = np.log(1/np.array(r))
    (D, b) = np.polyfit(x, y, deg=1)
    
    # search fit error value
    N = len(x)
    Sum = 0
    for i in range(N):
        Sum += (D*x[i] + b - y[i])**2
        
    errorfit = (1/N)*math.sqrt(Sum/(1+D**2))
    
    # figure size 10x5 inches
    plt.figure(1,figsize=(10,5)).canvas.set_window_title('Fractal Dimension Calculate')
    plt.subplots_adjust(left=0.04,right=0.98)
    plt.subplot(121)
    # plt.title(path)
    plt.imshow(image, cmap="gray")
    plt.axis('off')

    plt.subplot(122)  
    plt.title(f'Fractal dimension = {D:.3f} \n Fit Error = {errorfit:.3f}')
    
    plt.plot(x, y, 'ro',label='Calculated points')
    plt.plot(x, D*x+b, 'k--', label='Linear fit' )
    plt.legend(loc=4)
    plt.xlabel('log(1/r)')
    plt.ylabel('log(Nr)')
    plt.show()

