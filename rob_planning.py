import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from matplotlib.path import Path
import matplotlib.patches as patches

import imageio

import cv2


face = cv2.imread('world2.jpg')


fig, ax = plt.subplots()

# ax.imshow(data, cmap=cmap, norm=norm)
ax.imshow(face, alpha = 0.7)
# draw gridlines
# ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
# ax.set_xticks(np.arange(-0.5, 49.5, 1));
# ax.set_yticks(np.arange(-0.75, 49.5, 1));
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.yaxis.set_major_formatter(plt.NullFormatter())


rezolution = 25

face = cv2.resize(face, (rezolution,rezolution), interpolation = cv2.INTER_AREA)
face2d = face[::,::,0]

#%%




# data = np.random.rand(100, 100) * 1

data = np.zeros((rezolution,rezolution), dtype = float)

step = 1

face2d_r = face2d.reshape(rezolution**2,1)
data_r = data.reshape(rezolution**2,1)

for j in range(0,rezolution**2,1):
    
    if(face2d_r[j:j+step,::].mean() != 255):
        data_r[j:j+step,::] = 1
        # print(face2d_r[j:j+step,::].mean())
    
face = face2d_r.reshape(rezolution,rezolution)
data = data_r.reshape(rezolution,rezolution)
    
    
    
    

# create discrete colormap
cmap = colors.ListedColormap(['white', 'green'])
bounds = [0,0.5,1]
norm = colors.BoundaryNorm(bounds, cmap.N)




fig, ax = plt.subplots()
ax.imshow(data, cmap=cmap, norm=norm)
# ax.imshow(face, alpha = 0.7)


# draw gridlines
ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.5)
ax.set_xticks(np.arange(-0.5, 24.5, 1));
ax.set_yticks(np.arange(-0.5, 24.5, 1));
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.yaxis.set_major_formatter(plt.NullFormatter())




# ax.axis('off')

plt.show()




#%%






# fig = plt.figure()
# ax = fig.add_subplot(111)



# ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
# ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
# ax.axis('off') # removes the axis to leave only the shape



