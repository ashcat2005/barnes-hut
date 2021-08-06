# Barnes-Hut Algorithm

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class Node:
  '''
  Creation of a node.
  '''
  def __init__(self, mass, position, velocity):
    '''
    Inizialization of the object with mass and
    position coordinates (x,y)
    The attribute .child identifies if the node 
    represents a body (.child=None) or if the node 
    has child nodes.
    '''
    self.mass = mass
    
    #Position, velocity, acceleration
    self.position = position
    self.velocity=velocity
    self.accel= [0,0]
    
    # Mass product position. Usefull for the center of mass
    self.mr = mass*position 
    #Child
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
    new_node.position = new_node.mr / new_node.mass
    
    # (1) The body is localized in the appropiate quadrant using the 
    # method .quadrant_localization()
    quadrant = body.quadrant_localization()
    new_node.child[quadrant] = add_body(body, new_node.child[quadrant])
  return new_node

def Pointer(tree_node_acted, tree_node_acting, bodies):
    """
    Check the tree for any terminal nodes and finds it's acceleration
    then extract the node to a list
    """
    if tree_node_acted is not None:
        if tree_node_acted.child is None:
            Accelerator(tree_node_acted, tree_node_acting)
            bodies.append(tree_node_acted) 
        else:
            for i in range(4):
                Pointer(tree_node_acted.child[i],tree_node_acting, bodies)

def Accelerator(tree_node_acted , tree_node_acting):
    """
    Search for another node to calculate the force with.
    Calculates the total acceleration of each terminal node
    """
    if tree_node_acting is not None:
        Deltaxyz = tree_node_acted.position - tree_node_acting.position
        r = np.sqrt(np.sum(Deltaxyz*Deltaxyz))
        if r == 0:
            r = 0.0001
        if tree_node_acting.side/r > theta:      
            if tree_node_acting.child is None:
                tree_node_acted.accel-=G*Deltaxyz*tree_node_acting.mass/(r**3)
                #print(tree_node_acting.position,"--->",tree_node_acted.position)    
            else:
                for i in range(4):
                    Accelerator(tree_node_acted,tree_node_acting.child[i])
        else:
            #print(tree_node_acting.position,"--->",tree_node_acted.position, "  Skip")
            tree_node_acted.accel-=G*Deltaxyz*tree_node_acting.mass/(r**3)
                             
def Verlet(h, node_0, s0):
    '''
    ------------------------------------------
    Verlet(h, node_0, s0))
    ------------------------------------------
    Verlet method for solving a ODEs system.
    It needs two points to inizialize.
    ------------------------------------------
    Arguments:
    h: stepsize for the iteration
    node_0: initial configuration of particles
    s0: position of particles in the 
            (n-1)-step 
    ------------------------------------------
    Returns:
    Right hand side of the ODEs system as q1:
    q1[0] : x(n+1)
    q1[1] : y(n+1)
    q1[2] : vx(n)
    q1[3] : vy(n)
    ------------------------------------------
    '''
    q1 = np.zeros(2)
    v1 = np.zeros(2)
    q1[0] = 2*node_0.position[0] - s0[0] + node_0.accel[0]*h**2
    q1[1] = 2*node_0.position[1] - s0[1] + node_0.accel[1]*h**2
    v1[0] = (q1[0] - s0[0])/(2*h)
    v1[1] = (q1[1] - s0[1])/(2*h)
    return q1, v1

# N-BODIES SYSTEM DEFINITION #
# We will create a system with N-particles localized within a
# certain radius r_init centered at the point center.
# The level-0 quadrant corresponds to a square of side = 1.

G = 4*np.pi**2  # Gravitational constant
theta=0.5       # Condition to use the center of mass. (size/distance)

# Time Grid
dt = 1.E-4 # Time-step.
n = 10000 # Number of time-iterations.

"""Creation of the initial configuration"""

# The initial distribution is restricted to a circle of radius r_init.
r_init = 0.5
# Initial position of the center of the distribution
center = np.array([0.5, 0.5])

# Maximum initial velocity of the bodies.
v_init_max = 0.1 

# We will consider N-bodies with equal masses.
N = 5 # Number of bodies
masses = np.ones(N)

# Creation of the N-bodies.
np.random.seed(5) #Pseudo-random number generator for reproductibility

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
    bodies.append(Node(masses[i], init_pos[i], init_vel))

print('Total number of bodies: ', len(bodies))

"""First BHTree and Integration"""

## BHTree construction
root_node = None

for body in bodies:  
  body.reset_localization()
  root_node = add_body(body, root_node)
  
# Calculates the acceleration onto each body due to all external forces 
# and save it in a list
bodies=[]
Pointer(root_node, root_node, bodies)

#Graph
plt.figure(figsize=(8,8))
for body in bodies:
    plt.scatter(body.position[0], body.position[1], color='crimson', marker='.')
    
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.show()

#Backward Euler step to begin
for body in bodies:    
    S = np.zeros(2)
    S[0] = body.position[0] - dt*body.velocity[0] + 0.5*body.accel[0]*dt**2
    S[1] = body.position[1] - dt*body.velocity[1] + 0.5*body.accel[1]*dt**2
    #Verlet
    body.position, body.velocity = Verlet(dt, body, S)

plt.figure(figsize=(8,8))
for body in bodies:
    plt.scatter(body.position[0], body.position[1], color='crimson', marker='.')

plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.show()