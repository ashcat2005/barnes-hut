# Barnes-Hut Algorithm

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

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
    self.mass = mass
    self.position = position
    self.r_c = r_c
    # Mass times position. Useful to calculate center of mass
    self.mr = mass*position 
    self.position_c = position - center
    # Momentum 
    self.momentum = mass*velocity
    self.child = None
  
  def reset_localization(self):
    '''
    Resets the localization of the body to the 0th-order
    quadrant which has side = 1.
    '''
    self.side = 1.
    # Defines the relativew position w.r.t. the 0th-order 
    # quadrant. Note that we use copy in order to not alter 
    # the position attribute.
    self.relative_position = self.position.copy()

  def quadrant_localization(self):
    '''
    Localizes the node in the next level quadrant.
    Returns the number of the corresponding quadrant. 
    '''
    # Defines the size of the new quadrant
    self.side = 0.5*self.side 
    # Defines the relative position w.r.t the new quadrant
    self.relative_position = 2.*self.relative_position
    # A variable to store the location information
    quad = np.zeros(2, dtype=int)  
    #  Checks the location in both coordinates
    for i in range(2):
      if self.relative_position[i]<1.0:
        quad[i] = 0
      else:
        quad[i] = 1
        self.relative_position[i] = self.relative_position[i] - 1.
    return quad[1] + 2*quad[0]


# N-BODIES SYSTEM DEFINITION #
# We will create a system with N-particles localized within a
# certain radius r_init centered at the point center.
# The level-0 quadrant corresponds to a square of side = 1.

G = 4*np.pi**2 # Gravitational constant

# Time Grid
dt = 1.E-3 # Time-step.
n = 10000 # Number of time-iterations.

# The initial distribution is restricted to a circle of radius r_init.
r_init = 0.1
# Initial position of the center of the distribution
center = np.array([0.5, 0.5])

# Maximum initial velocity of the bodies.
v_init_max = 0.1 


# We will consider N-bodies with equal masses.
N = 50 # Number of bodies
masses = np.ones(N)

# Creation of the N-bodies.
np.random.seed(1) #Pseudo-random number generator for reproductibility

# (x,y) position radomly generated in a square with side 2*r_init
init_pos = np.zeros([N,2])
init_pos[:,0] = np.random.random(N)*2.*r_init + center[0] - r_init
init_pos[:,1] = np.random.random(N)*2.*r_init + center[1] - r_init

# Position and distance w.r.t. the center
init_pos_c = init_pos[:]-center
r_c = np.linalg.norm(init_pos_c, axis=1)


bodies = [] # List of bodies in the system

# Loop for creating the bodies
for i in range(N):
  # Only the bodies inside a circle of radius r_init are considered.
  if r_c[i] < r_init:
    # Initial velocity, proportional to the disctance from the center
    init_vel = np.array([-init_pos_c[i,1], init_pos_c[i,0]])*\
               v_init_max*(r_c[i]/r_init)
    # Adding the body to the list
    bodies.append(Node(masses[i], init_pos[i], init_vel, r_c[i]))

print('Total number of bodies: ', len(bodies))


def add_body(body, tree_node):
  '''
  This function defines the BHTree using three conditions
  for each body.
  '''
  # Minimum allowed size for the qwuadrants
  minimum_quadrant_side = 1.E-2

  # Condition 1. If tree_node does not contain a body, 
  # the body is put in there.
  if tree_node is None:
    new_node = body 
  else:
    new_node = None
  
  if tree_node is not None and tree_node.side > minimum_quadrant_side:
    # Condition 2. If tree_node is an external node (i.e. already 
    # contains a body but has no child) we create the 4 children
    # and locate the bodies.
    if tree_node.child is None:
      # coopy the tree_node info
      new_node = deepcopy(tree_node)
      # Create 4 children (with value None)
      new_node.child = [None for i in range(4)]
      # Insert the present body in the appropiate quadrant using the
      # method .quadrant_localization()
      quadrant = tree_node.quadrant_localization()
      new_node.child[quadrant] = tree_node
      # The second body is localized in the corresponding quadrant in (1)
    else:
      # Condition 3. If  tree_node is an internal node (i.e. it already
      # has children), we do not modify the children. The body will be 
      # located in the appropiate quadrant in (1)
      new_node = tree_node
      
    # In any of the cases, the mass and position of the center of mass
    # for the new node are updated
    new_node.mass += body.mass 
    new_node.mr += body.mr
    # (1) The body is localized in the appropiate quadrant using the 
    # method .quadrant_localization()
    quadrant = body.quadrant_localization()
    new_node.child[quadrant] = add_body(body, new_node.child[quadrant])
  return new_node


## First BHTree construction
root_node = None

for body in bodies:
  body.reset_localization()
  root_node = add_body(body, root_node)
  print(body, root_node)

plt.figure(figsize=(8,8))
for body in bodies:
  plt.scatter(body.position[0], body.position[1], color='crimson', marker='.')

plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Initial configuration of the system')
plt.savefig('InitialConfiguration.jpg')
plt.show()