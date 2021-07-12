# Barnes-Hut Algorithm

import numpy as np
import matplotlib.pyplot as plt

class Node:
  '''
  Creation of a node.
  '''
  def __init__(self, mass, position, velocity, r_c):
    '''
    Inizialization of the object with mass and
    position coordinates (x,y)
    The attribute .child identifies if the node 
    represents a body (.child=None) or if the node 
    has child nodes.
    '''
    self.m = mass
    self.position = position
    self.r_c = r_c
    # Mass times position. Useful to calculate center of mass
    self.mr = mass*position 
    self.position_c = position - center
    # Momentum 
    self.momentum = mass*velocity
    self.child = None


# MAIN

G = 4*np.pi**2 # Gravitational constant

# We will consider N-bodies with equal masses.
mass = 1.0
N = 1000 # Number of bodies
v_init_max = 0.1 # Maximum initial velocity of the bodies.

# Time Grid
dt = 1.E-3 # Time-step.
n = 10000 # Number of time-iterations.

# The initial distribution is restricted to a circle of radius r_init.
r_init = 0.1
# Initial position of the center of the distribution
center = np.array([0.4, 0.4])

# Creation of the N-bodies.
np.random.seed(1) #Pseudo-random number generator for reproductibility

# (x,y) position radomly generated in a square with side 2*r_init
init_pos = np.zeros([N,2])
init_pos[:,0] = np.random.random(N)*2.*r_init + center[0] - r_init
init_pos[:,1] = np.random.random(N)*2.*r_init + center[1] - r_init

# Position and distance w.r.t. the center
init_pos_c = init_pos[:]-center
r_c = np.linalg.norm(init_pos_c, axis=1)

bodies = []
# Only the bodies inside a circle of radius r_init are considered.
for i in range(N):
  if r_c[i] < r_init:
    # Initial velocity, proportional to the disctance from the center
    init_vel = np.array([-init_pos_c[i,1], init_pos_c[i,0]])*\
               v_init_max*(r_c[i]/r_init)
    bodies.append(Node(mass, init_pos[i], init_vel, r_c[i]))


print('Total number of bodies: ', len(bodies))






plt.figure(figsize=(8,8))
for body in bodies:
  plt.scatter(body.position[0], body.position[1], color='crimson', marker='.')

plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.show()