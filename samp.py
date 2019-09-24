from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
#x, y, z, v = (np.random.randint(4,10))
x = np.random.randint(15,size = 3)
y = np.random.randint(15,size = 3)
z = np.random.randint(15,size = 3)
v=np.random.randint(100,size=3)
c = np.abs(v)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("hot")
cax = ax.scatter(x, y, z, v, s=50, c=c, cmap=cmhot)
print(x,y,z)
print(v)
plt.show()
