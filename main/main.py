# Barnes-Hut Algorithm



import numpy as np
import matplotlib.pyplot as plt

class Node:
  '''
  Creation of a node.
  '''
  def __init__(self, mass, position):
    '''
    Inizialization of the object with mass and
    position coordinates (x,y)
    The attribute .child identifies if the node 
    represents a body (.child=None) or if the node 
    has child nodes.
    '''
    self.m = mass
    self.r = position
    # mass times position. Useful to calculate center of mass
    self.mr = mass*position 
    self.child = None



# MAIN

G = 4*np.pi**2 # Gravitational constant

# We will consider N-bodies with equal masses.
mass = 1.0
N = 1000 # Number of bodies
v_max_init = 0.1 # Maximum initial velocity of the bodies.

# Time Grid
dt = 1.E-3 # Time-step.
n = 10000 # Number of time-iterations.

# The initial distribution is restricted to a circle of radius r_init.
r_init = 0.1
# Initial position of the center of the distribution
x_c = 0.4
y_c = 0.4

# Creation of the N-bodies.
np.random.seed(1) #Pseudo-random number generator

# (x,y) position generated in a square with side 2*r_init
x = np.random.random(N) *2.*r_init + x_c - r_init
y = np.random.random(N) *2.*r_init + y_c - r_init

# Only the bodies inside a circle of radius r_init are considered.
bodies = [ Node(mass, np.array([px, py])) for (px,py) in zip(x, y) \
               if (px-x_c)**2 + (py-y_c)**2 < r_init**2 ]

print('Total number of bodies: ', len(bodies))

plt.figure(figsize=(8,8))
for body in bodies:
  plt.scatter(body.r[0], body.r[1], color='crimson', marker='.')

plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.show()