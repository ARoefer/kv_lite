# KV-Lite - Kineverse 2.0

This package is the second attempt at implementing the *Kineverse* articulation framework.

 - [Original Website](http://kineverse.cs.uni-freiburg.de/)
 - [Paper](https://arxiv.org/abs/2012.05362)

## Installation

The package is implemented as a ROS-package as it is mostly meant for the use in the robotics context, however, as of now (20th June 2023), it does not have a hard ROS dependency.

Clone the repository - preferably to a ROS workspace - and install the requirements using `pip`:

```bash
# In kv_lite
pip install -r requirements.txt
```

In the case of ROS, build your workspace, reload it and you're good to go.

### Additional Dependencies

Some of the examples require additional packages. Currently it is

 - [roebots](https://github.com/ARoefer/roebots) a package collecting some utility functions for working with robotics. Used here for resolving ROS package paths without a running ROS core.


## Usage

### Preamble

*KV-lite* is a framework for building and managing kinematics of articulated objects (doors, drawers, robots, general mechanisms). The aim is to be more flexible in the range of possible articulations than frameworks such as URDF, or SDF. These structures are managed as graphs, which can be compiled into analytical, differentiable forward-kinematic expressions. The expressions can be queried for their arguments, often also referred to as *(free) symbols*, for which a KV-lite model can hold constraints. In the end, these constraints and expressions can be used in any algorithm.

The general promise of KV-lite/Kineverse is: *If there is a closed-form expression for your articulation, you can use it.*

### Concepts

Before going to full models of articulated structures, we need to take a look at KV-lite's basic building blocks. Fundamentally, there are only four things:

 - Symbols: The variables of expressions. Implemented as `KVSymbol`.
 - Expressions: Some mathematical expression. Implemented as `KVExpr`.
 - Matrices: A structured arrangement of expressions. Implemented as `KVArray` on top of `numpy`'s arrays.
 - Constraints: Triples of an expressions `lb, ub, e` which expresses the constraint `lb <= e <= ub`. Implemented simply as `Constraint`.

Let us look at an example of using these:

```python
from kv_lite import gm  # GM -> Gradient math (legacy term)

# Creating our first symbol
a = gm.KVsymbol('a')

# We can use a normally to build up symbolic expressions
e1 = a * 4
print(e1)  # >> (4*a)

# We have to use the symbolic functions from gm
e2 = gm.cos(e1)
print(e2)  # >> cos((4*a))

# We can inspect our expressions
print(e2.is_symbolic)  # >> True
print(e2.symbols)      # >> frozenset({KV(a)})

# Constant expressions also exist
e_constant = gm.KVExpr(4) * gm.KVExpr(5)
print(e_constant)              # >> KV(20)
print(e_constant.is_symbolic)  # >> False
print(e_constant.symbols)      # >> frozenset()
# Non-symbolic expressions can be turned directly into float
print(float(e_constant))       # >> 20.0

# We can evaluate expressions by assigning values to their symbols
print(e1.eval({a: 3}))   # >> 12.0
print(e2.eval({a: 2}))   # >> -0.1455...
print(e_constant.eval()) # >> 20.0

# Note two things:
#  1. All expressions always evaluate to float
#  2. eval() only filters for expected variables:

print(e1.eval({a: 3, gm.KVSymbol('b'): 2}))  # >> 12.0

# We can easily generate the jacobian of an expression
print(e1.jacobian([a]))  # >> [[KV(4)]]
print(e2.jacobian([a]))  # >> [[KV(@1=4, (-(@1*sin((@1*a)))))]]
```

In the last stages of the code above, we generate instances of `KVArray`. As stated before, this is the array implementation of KV-lite. It is a small extension of `numpy`'s `ndarray` type and thus supports all typical numpy array operations, such as indexing, slicing, stacking, and whatever else. All functions provided in KV-lite are vectorized, meaning that they broadcast across arrays. *Careful*: When given a container, KV-lite operations will **always** return a `KVArray`. Let us look at a couple of examples of using `KVArray`:

```python
from kv_lite import gm

# Let's create a couple more symbols for ourselves
a, b, c, d = [gm.KVSymbol(x) for x in 'abcd']

# CAREFUL: The left-hand side determines the array type, thus we create a KVArray here
m = gm.diag([1, 2, 3]) # Equivalent to gm.KVArray(np.diag([1, 2, 3]))
v = gm.KVArray([a, b, c]).reshape((3, 1))

r = m.dot(v)  # @ operator would work too
print(r) 
# [[KV(a)]
#  [KV((2*b))]
#  [KV((3*c))], dtype=object]

# KVArray's offer the same introspection as expressions
print(r.is_symbolic)  # >> True
print(r.symbols)      # >> frozenset({KV(c), KV(b), KV(a)})
print(r.jacobian([a, b, c]))
# [[KV(1), KV(0), KV(0)]
#  [KV(0), KV(2), KV(0)]
#  [KV(0), KV(0), KV(3)]]

j_r = r.jacobian([a, b, c])

# Typical numpy indexing works as expected
print(j_r.T)
# [[KV(1) KV(0) KV(0)]
#  [KV(0) KV(2) KV(0)]
#  [KV(0) KV(0) KV(3)]]
print(j_r[ 1,   0])
# 0
print(j_r[ 0])
# [KV(1) KV(0) KV(0)]
print(j_r[ :,   2])
# [KV(0) KV(0) KV(3)]
print(j_r[:2, 2:3])
# [[KV(0)]
#  [KV(0)]]

# Jacobian wrt foreign symbols is 0
print(r.jacobian([d]))

```

### Transformations & Spatial Types

Creating arbitrary matrices is fine and dandy, but also offers a growing set of implementations for typical spatial entities. By default, KV-lite uses 4x4 homogenous transforms as base representation for entities from SO(3) and SE(3). Let us look at a quick run-down of the options

```python
import numpy as np
from kv_lite import gm

# Create a homogeneous vector -> Not affected by translation
v = gm.vector3(1, 2, 3)
print(v)
# [[1]
#  [2]
#  [3]
#  [0]]

# L2 norm of a vector
print(gm.norm(v))  # >> 3.74165738...

# Create a homogeneous point -> Affected by translation and rotation
p = gm.point3(1, 2, 3)
print(p)
# [[1]
#  [2]
#  [3]
#  [1]]

# Create an identity transform
identity = gm.Transform.identity()

# A pure linear translation along the x axis by two meters
trans1 = gm.Transform.from_xyz(2, 0, 0)

# A 90 degree rotation around the Y-axis
rot1   = gm.Transform.from_euler(0, 0, np.deg2rad(90))

# A 45 degree rotation around v
rot2   = gm.Transform.from_axis_angle(v / gm.norm(v), np.deg2rad(45))

# An identity rotation from a quaternion
rot3   = gm.Transform.from_quat(0, 0, 0, 1)

# A combined rotation and translation
# The same exists for from_xyz_aa, from_xyz_quat
tf1    = gm.Transform.from_xyz_euler(2, 0, 0, 0, 0, np.deg2rad(90))

# Use Transform.inverse to invert homogeneous transformations
tf1_inv = gm.Transform.inverse(tf1)

# gm.Transform also provides some structured introspection into transforms
rot4   = gm.Transform.rot(tf1_inv)   # Generate a pure rotation transform from tf1_inv
trans2 = gm.Transform.trans(tf1_inv) # Generate a pure translation transform from tf1_inv

tf1_inv_x = gm.Transform.x(tf1_inv)  # X-column of transform
tf1_inv_y = gm.Transform.y(tf1_inv)  # Y-column of transform
tf1_inv_z = gm.Transform.z(tf1_inv)  # Z-column of transform
tf1_inv_w = gm.Transform.w(tf1_inv)  # W-column of transform - identical to gm.Transform.pos(tf1_inv)

# Of course, transform creation also works with symbols!
a, b = [gm.KVSymbol(x) for x in 'ab']

# A transform translating by 2a along X and rotating by b around Z
tf2  = gm.Transform.from_xyz_euler(2*a, 0, 0, 0, b, 0)
print(tf2)
# [[KV(cos(b)) KV(0) KV(sin(b)) KV((2*a))]
#  [KV(0) KV(1) KV(0) KV(0)]
#  [KV((-sin(b))) KV(0) KV(cos(b)) KV(0)]
#  [KV(0) KV(0) KV(0) KV(1)]]

# Transformations are chained by matrix multiplication
tf3 = tf1.dot(tf2)  # @ would work as well
```

### Symbol Typing

So far, we have always created all symbols using `KVSymbol`. This is fine for a general use of KV-lite and its symbolic math functionality, but to use its full modelling capabilities we must understand symbol typing. In KV-lite, it is possible to create symbols of certain types: Positions, Velocities, Accelerations, Jerks, and Snaps. These symbols behave normally, but have the additional feature, that they can be differentiated and integrated. Using this system, it is possible to create different constraints for the position, velocity, etc. of one degree of freedom of a model. Additionally, we can generate the tangent expression of an expression, which models the rate of change of the function at a given point, given the derivatives of it's symbols.
Let's make this topic a bit more approachable:

```python
from kv_lite import gm

# We create a few symbols modelling the position of degrees of freedom a, b, c
a, b, c = [gm.Position(x) for x in 'abc']
print(a, b, c)  # >> a__position b__position c__position

# We can generate the symbols referencing the derivative and the integral of a typed symbol
print(a.derivative())             # >> a__velocity
print(a.derivative().integral())  # >> a__position

# We cannot integrate beyond position or differentiate beyond snap
try:
    a.integral()
except RuntimeError as e:
    print(e)  # >> Cannot integrate symbol beyond position.

# Let us create an expression
e1 = a * 4 -b
print(e1)  # >> ((4*a__position)-b__position)

# We can now not just generate the jacobian wrt a__position,
# but also generate the tangent expression
print(e1.tangent())  # >> ((4*a__velocity)-b__velocity)

# Note: A typed symbol's tangent is the symbol of its derivation
print(a.tangent())  # >> a__velocity
```

### Models
