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

Creating arbitrary matrices is fine and dandy, but also offers a growing set of implementations for typical spatial entities. By default, KV-lite uses 4x4 homogenous transforms as base representation for entities from SO(3) and SE(3).

```python
from kv_lite import gm
```
