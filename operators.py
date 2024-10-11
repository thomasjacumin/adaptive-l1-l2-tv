import numpy as np
from scipy import sparse
from scipy.sparse import bmat

import quadmesh

def x_derivative_forward(view, bc = "N"):
  Av = []
  Ax = []
  Ay = []
  elements = view.getElements()
  dofs = len(elements)
  for n in elements:
    eastElements = view.getEastNeighbours(n)
    if len(eastElements) == 0: # On the boundary
      if bc == "N":
        Ax.append(n.indice)
        Ay.append(n.indice)
        Av.append(0)
      else:
        Ax.append(n.indice)
        Ay.append(n.indice)
        Av.append(-1./n.dx)
    else:
      if len(eastElements) == 2: # Dangling 1
        Ax.append(n.indice)
        Ay.append(n.indice)
        Av.append(-4./(3*n.dx))
        #
        Ax.append(n.indice)
        Ay.append(eastElements[0].indice)
        Av.append(2./(3*n.dx))
        #
        Ax.append(n.indice)
        Ay.append(eastElements[1].indice)
        Av.append(2./(3*n.dx))
      else:
        if eastElements[0].level == n.level: # regular
          Ax.append(n.indice)
          Ay.append(n.indice)
          Av.append(-1./n.dx)
          #
          Ax.append(n.indice)
          Ay.append(eastElements[0].indice)
          Av.append(1./n.dx)
        else:
          if n.parent.children[quadmesh.NORTH_EAST] == n: # dangling 3
            southElements = view.getSouthNeighbours(n)
            Ax.append(n.indice)
            Ay.append(n.indice)
            Av.append(-1./(3*n.dx))
            #
            Ax.append(n.indice)
            Ay.append(southElements[0].indice)
            Av.append(-1./(3*n.dx))
            #
            Ax.append(n.indice)
            Ay.append(eastElements[0].indice)
            Av.append(2./(3*n.dx))
          elif n.parent.children[quadmesh.SOUTH_EAST] == n: # dangling 2
            northElements = view.getNorthNeighbours(n)
            Ax.append(n.indice)
            Ay.append(n.indice)
            Av.append(-1./(3*n.dx))
            #
            Ax.append(n.indice)
            Ay.append(northElements[0].indice)
            Av.append(-1./(3*n.dx))
            #
            Ax.append(n.indice)
            Ay.append(eastElements[0].indice)
            Av.append(2./(3*n.dx))
          else:
            print("error")
  return sparse.csr_matrix((Av, (Ax, Ay)), shape=[dofs, dofs])

def x_derivative_backward(view, bc = "N"):
  Av = []
  Ax = []
  Ay = []
  elements = view.getElements()
  dofs = len(elements)
  for n in elements:
    westElements = view.getWestNeighbours(n)
    if len(westElements) == 0: # on the left-boundary
      if bc == "N":
        Ax.append(n.indice)
        Ay.append(n.indice)
        Av.append(0)
      else:
        Ax.append(n.indice)
        Ay.append(n.indice)
        Av.append(1./n.dx)
    else:
      if len(westElements) == 2: # dangling 1
        eastElements = view.getEastNeighbours(n)
        if len(eastElements) == 0: # on the right-boundary
          Ax.append(n.indice)
          Ay.append(westElements[0].indice)
          Av.append(-2./(3*n.dx))
          #
          Ax.append(n.indice)
          Ay.append(westElements[1].indice)
          Av.append(-2./(3*n.dx))
        else:
          Ax.append(n.indice)
          Ay.append(n.indice)
          Av.append(4./(3*n.dx))
          #
          Ax.append(n.indice)
          Ay.append(westElements[0].indice)
          Av.append(-2./(3*n.dx))
          #
          Ax.append(n.indice)
          Ay.append(westElements[1].indice)
          Av.append(-2./(3*n.dx))
      else:
        if westElements[0].level == n.level: # regular
          eastElements = view.getEastNeighbours(n)
          if len(eastElements) == 0: # on the right-boundary
            Ax.append(n.indice)
            Ay.append(westElements[0].indice)
            Av.append(-1./n.dx)
          else:
            Ax.append(n.indice)
            Ay.append(n.indice)
            Av.append(1./n.dx)
            #
            Ax.append(n.indice)
            Ay.append(westElements[0].indice)
            Av.append(-1./n.dx)
        else:
          if n.parent.children[quadmesh.NORTH_WEST] == n: # dangling 3
            southElements = view.getSouthNeighbours(n)
            Ax.append(n.indice)
            Ay.append(n.indice)
            Av.append(1./(3*n.dx))
            #
            Ax.append(n.indice)
            Ay.append(southElements[0].indice)
            Av.append(1./(3*n.dx))
            #
            Ax.append(n.indice)
            Ay.append(westElements[0].indice)
            Av.append(-2./(3*n.dx))
          elif n.parent.children[quadmesh.SOUTH_WEST] == n: # dangling 2
            northElements = view.getNorthNeighbours(n)
            Ax.append(n.indice)
            Ay.append(n.indice)
            Av.append(1./(3*n.dx))
            #
            Ax.append(n.indice)
            Ay.append(northElements[0].indice)
            Av.append(1./(3*n.dx))
            #
            Ax.append(n.indice)
            Ay.append(westElements[0].indice)
            Av.append(-2./(3*n.dx))
          else:
            print("error")

  return sparse.csr_matrix((Av, (Ax, Ay)), shape=[dofs, dofs])

def y_derivative_forward(view, bc = "N"):
  Av = []
  Ax = []
  Ay = []
  elements = view.getElements()
  dofs = len(elements)
  for n in elements:
    southElements = view.getSouthNeighbours(n)    
    if len(southElements) == 0: # On the boundary
      if bc == "N":
        Ax.append(n.indice)
        Ay.append(n.indice)
        Av.append(0)
      else:
        Ax.append(n.indice)
        Ay.append(n.indice)
        Av.append(-1./n.dy)
    else:
      if len(southElements) == 2: # dangling 1
        Ax.append(n.indice)
        Ay.append(n.indice)
        Av.append(-4./(3*n.dy))
        #
        Ax.append(n.indice)
        Ay.append(southElements[0].indice)
        Av.append(2./(3*n.dy))
        #
        Ax.append(n.indice)
        Ay.append(southElements[1].indice)
        Av.append(2./(3*n.dy))
      else:
        if southElements[0].level == n.level: # regular
          Ax.append(n.indice)
          Ay.append(n.indice)
          Av.append(-1./n.dy)
          #
          Ax.append(n.indice)
          Ay.append(southElements[0].indice)
          Av.append(1./n.dy)
        else:
          if n.parent.children[quadmesh.SOUTH_WEST] == n: # dangling 2
            eastElements = view.getEastNeighbours(n)
            Ax.append(n.indice)
            Ay.append(n.indice)
            Av.append(-1./(3*n.dy))
            #
            Ax.append(n.indice)
            Ay.append(eastElements[0].indice)
            Av.append(-1./(3*n.dy))
            #
            Ax.append(n.indice)
            Ay.append(southElements[0].indice)
            Av.append(2./(3*n.dy))
          elif n.parent.children[quadmesh.SOUTH_EAST] == n: # dangling 3
            westElements = view.getWestNeighbours(n)
            Ax.append(n.indice)
            Ay.append(n.indice)
            Av.append(-1./(3*n.dy))
            #
            Ax.append(n.indice)
            Ay.append(westElements[0].indice)
            Av.append(-1./(3*n.dy))
            #
            Ax.append(n.indice)
            Ay.append(southElements[0].indice)
            Av.append(2./(3*n.dy))
          else:
            print("error")
  return sparse.csr_matrix((Av, (Ax, Ay)), shape=[dofs, dofs])

def y_derivative_backward(view, bc = "N"):
  Av = []
  Ax = []
  Ay = []
  elements = view.getElements()
  dofs = len(elements)
  for n in elements:
    northElements = view.getNorthNeighbours(n)  
    if len(northElements) == 0: # on the top-boundary
      if bc == "N":
        Ax.append(n.indice)
        Ay.append(n.indice)
        Av.append(0)
      else:
        Ax.append(n.indice)
        Ay.append(n.indice)
        Av.append(1./n.dy)
    else:
      if len(northElements) == 2: # dangling 1
        southElements = view.getSouthNeighbours(n) 
        if len(southElements) == 0:  # on the bottom-boundary
          Ax.append(n.indice)
          Ay.append(northElements[0].indice)
          Av.append(-2./(3*n.dy))
          #
          Ax.append(n.indice)
          Ay.append(northElements[1].indice)
          Av.append(-2./(3*n.dy))
        else:
          Ax.append(n.indice)
          Ay.append(n.indice)
          Av.append(4./(3*n.dy))
          #
          Ax.append(n.indice)
          Ay.append(northElements[0].indice)
          Av.append(-2./(3*n.dy))
          #
          Ax.append(n.indice)
          Ay.append(northElements[1].indice)
          Av.append(-2./(3*n.dy))
      else:
        if northElements[0].level == n.level: # regular
          southElements = view.getSouthNeighbours(n) 
          if len(southElements) == 0:  # on the bottom-boundary
            Ax.append(n.indice)
            Ay.append(northElements[0].indice)
            Av.append(-1./n.dy)
          else:
            Ax.append(n.indice)
            Ay.append(n.indice)
            Av.append(1./n.dy)
            #
            Ax.append(n.indice)
            Ay.append(northElements[0].indice)
            Av.append(-1./n.dy)
        else:
          if n.parent.children[quadmesh.NORTH_WEST] == n: # dangling 2
            eastElements = view.getEastNeighbours(n)
            Ax.append(n.indice)
            Ay.append(n.indice)
            Av.append(1./(3*n.dy))
            #
            Ax.append(n.indice)
            Ay.append(eastElements[0].indice)
            Av.append(1./(3*n.dy))
            #
            Ax.append(n.indice)
            Ay.append(northElements[0].indice)
            Av.append(-2./(3*n.dy))
          elif n.parent.children[quadmesh.NORTH_EAST] == n: # dangling 3
            westElements = view.getWestNeighbours(n)
            Ax.append(n.indice)
            Ay.append(n.indice)
            Av.append(1./(3*n.dy))
            #
            Ax.append(n.indice)
            Ay.append(westElements[0].indice)
            Av.append(1./(3*n.dy))
            #
            Ax.append(n.indice)
            Ay.append(northElements[0].indice)
            Av.append(-2./(3*n.dy))
          else:
            print("error")

  return sparse.csr_matrix((Av, (Ax, Ay)), shape=[dofs, dofs])

def gradOperator(view, bc = "N"):
  x = x_derivative_forward(view, bc)
  y = y_derivative_forward(view, bc)
  return  bmat([ [x], [y] ])


def divOperator(view, bc = "N"):
  x = x_derivative_backward(view, bc)
  y = y_derivative_backward(view, bc)
  return  bmat([ [x, y] ])

def gradOperator_b(view, bc = "N"):
  x = x_derivative_backward(view, bc)
  y = y_derivative_backward(view, bc)
  return  bmat([ [x], [y] ])

def gradOperator_centered(mesh, bc = "N"):
    return 0.5*(gradOperator(mesh, bc) + gradOperator_b(mesh, bc))
