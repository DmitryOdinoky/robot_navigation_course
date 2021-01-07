import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from matplotlib.path import Path
import matplotlib.patches as patches

import imageio

import cv2
import tqdm


face = cv2.imread('world2.jpg')


fig, ax = plt.subplots()


ax.imshow(face, alpha = 0.7)
# draw gridlines
# ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
# ax.set_xticks(np.arange(-0.5, 49.5, 1));
# ax.set_yticks(np.arange(-0.75, 49.5, 1));
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.yaxis.set_major_formatter(plt.NullFormatter())

# global params
rezolution = 25
step = 1

# create discrete colormap
cmap = colors.ListedColormap(['white', 'green'])
bounds = [0,0.5,1]
norm = colors.BoundaryNorm(bounds, cmap.N)

# resize image to our selected resolution
face = cv2.resize(face, (rezolution,rezolution), interpolation = cv2.INTER_AREA)

# take just first layer (single 2d matrix)
face2d = face[::,::,0]

#%%


fig, ax = plt.subplots()


data = np.zeros((rezolution,rezolution), dtype = float)


# reshape stuff, easier to iterate over array 
face2d_r = face2d.reshape(rezolution**2,1)
data_r = data.reshape(rezolution**2,1)

for j in range(0,rezolution**2,1):
    
    if(face2d_r[j:j+step,::].mean() != 255):
        data_r[j:j+step,::] = 1
        # print(face2d_r[j:j+step,::].mean())


def plot_point(x,y):
    
     plt.scatter(x-0.5, y-0.5, sizes=[100,100], facecolor='blue')
     ax.annotate((x, y), xy =(x, y), 
     xytext =(x, y))
        
for j in range(0,rezolution**2-rezolution,1):
        
    # resore original indexes by indexing reshaped array
    index = np.unravel_index(np.ravel_multi_index((j, 0), face2d_r.shape), face2d.shape)
    
    if(((data_r[j-rezolution] == 0) and (data_r[j-1] == 0) and (data_r[j] != 0))):        
        plot_point(index[1],index[0]) 
    elif(((data_r[j+rezolution] == 0)) and (data_r[j+1] == 0) and (data_r[j] != 0)):
        plot_point(index[1]+1,index[0]+1)
    elif(((data_r[j+rezolution] == 0)) and (data_r[j+1] == 0) and (data_r[j] != 0)):
        plot_point(index[1]+1,index[0]+1)
    elif(((data_r[j+rezolution] == 0) and (data_r[j-1] == 0) and (data_r[j] != 0))):        
        plot_point(index[1],index[0]+1)
    elif(((data_r[j-rezolution] == 0) and (data_r[j+1] == 0) and (data_r[j] != 0))):     
        plot_point(index[1]+1,index[0])
        
face = face2d_r.reshape(rezolution,rezolution)
data = data_r.reshape(rezolution,rezolution)
    
ax.imshow(data, cmap=cmap, norm=norm)    
    


#%%    


# plt.ylim(10,25)


# draw gridlines
ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.5)
ax.set_xticks(np.arange(-0.5, rezolution-0.5, 1));
ax.set_yticks(np.arange(-0.5, rezolution-0.5, 1));
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.yaxis.set_major_formatter(plt.NullFormatter())




# ax.axis('off')

plt.show()







