import numpy as np
from scipy import sparse

def gradOperator(view, bc = "N"):
  Av = []
  Ax = []
  Ay = []
  elements = view.getElements()
  dofs = len(elements)
  for n in elements:
    # Dérivée en x:
    ## On est sur le bord droit
    eastElements = view.getEastNeighbours(n)
    westElements = view.getWestNeighbours(n)
    northElements = view.getNorthNeighbours(n)
    southElements = view.getSouthNeighbours(n)
    if len(eastElements) == 0:
      if bc == "N":
        Ax.append(n.indice)
        Ay.append(n.indice)
        Av.append(0)
      else:
        Ax.append(n.indice)
        Ay.append(n.indice)
        Av.append(-1./n.dx)
    else:
      # Le noeud à droite est 2x plus petit:
      if len(eastElements) == 2:
        # Dangling 1
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
        # Le noeud à droite est de la même taille
        if eastElements[0].dx == n.dx:
          # Regular
          Ax.append(n.indice)
          Ay.append(n.indice)
          Av.append(-1./n.dx)
          #
          Ax.append(n.indice)
          Ay.append(eastElements[0].indice)
          Av.append(1./n.dx)
        # Le noeud à droite est 2x plus grand:
        else:
          # print("dx i="+str(n.indice))
          # le noeud est au nord
          if n.y <= eastElements[0].y:
            # print(n.indice)
            # print("N")
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
          # le noeud est au sud
          else:
            # print("S")
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
            
    
    # Dérivée en y:
    ## On est sur le bord en bas
    if len(southElements) == 0:
      if bc == "N":
        Ax.append(dofs + n.indice)
        Ay.append(n.indice)
        Av.append(0)
      else:
        Ax.append(dofs + n.indice)
        Ay.append(n.indice)
        Av.append(-1./n.dy)
    else:
      # Le noeud en bas est 2x plus petit:
      if len(southElements) == 2:
        # Dangling 1
        Ax.append(dofs + n.indice)
        Ay.append(n.indice)
        Av.append(-4./(3*n.dy))
        #
        Ax.append(dofs + n.indice)
        Ay.append(southElements[0].indice)
        Av.append(2./(3*n.dy))
        #
        Ax.append(dofs + n.indice)
        Ay.append(southElements[1].indice)
        Av.append(2./(3*n.dy))
      else:
        # Le noeud en bas est de la même taille
        if southElements[0].dx == n.dx:
          # Regular
          Ax.append(dofs + n.indice)
          Ay.append(n.indice)
          Av.append(-1./n.dy)
          #
          Ax.append(dofs + n.indice)
          Ay.append(southElements[0].indice)
          Av.append(1./n.dy)
        # Le noeud en bas est 2x plus grand:
        else:
          # print("dy i="+str(n.indice))
          # le noeud est à l'ouest
          if n.x <= southElements[0].x:
            # print("W")
            Ax.append(dofs + n.indice)
            Ay.append(n.indice)
            Av.append(-1./(3*n.dy))
            #
            Ax.append(dofs + n.indice)
            Ay.append(eastElements[0].indice)
            Av.append(-1./(3*n.dy))
              #
            Ax.append(dofs + n.indice)
            Ay.append(southElements[0].indice)
            Av.append(2./(3*n.dy))
          # le noeud est à l'est
          else:
            # print("E")
            Ax.append(dofs + n.indice)
            Ay.append(n.indice)
            Av.append(-1./(3*n.dy))
            #
            Ax.append(dofs + n.indice)
            Ay.append(westElements[0].indice)
            Av.append(-1./(3*n.dy))
              #
            Ax.append(dofs + n.indice)
            Ay.append(southElements[0].indice)
            Av.append(2./(3*n.dy))
    
  A = sparse.csr_matrix((Av, (Ax, Ay)), shape=[2*dofs, dofs])
  return A # cupyx.scipy.sparse.csr_matrix(A)

# def divOperator(mesh, bc = "N"):
#   return -gradOperator(mesh, bc).transpose()

def divOperator(view, bc = "N"):
  Av = []
  Ax = []
  Ay = []
  elements = view.getElements()
  dofs = len(elements)
  for n in elements:
    # Dérivée en x:
    ## On est sur le bord droit
    eastElements = view.getEastNeighbours(n)
    westElements = view.getWestNeighbours(n)
    northElements = view.getNorthNeighbours(n)
    southElements = view.getSouthNeighbours(n)
    # Dérivée en x:
    ## On est sur le bord droit
    if len(westElements) == 0:
      if bc == "N":
        Ax.append(n.indice)
        Ay.append(n.indice)
        Av.append(0)
      else:
        Ax.append(n.indice)
        Ay.append(n.indice)
        Av.append(-1./n.dx)
    else:
      # Le noeud à gauche est 2x plus petit:
      if len(westElements) == 2:
        # Dangling 1
        Ax.append(n.indice)
        Ay.append(n.indice)
        Av.append(-4./(3*n.dx))
        #
        Ax.append(n.indice)
        Ay.append(westElements[0].indice)
        Av.append(2./(3*n.dx))
        #
        Ax.append(n.indice)
        Ay.append(westElements[1].indice)
        Av.append(2./(3*n.dx))
      else:
        # Le noeud à gauche est de la même taille
        if westElements[0].dx == n.dx:
          # Regular
          Ax.append(n.indice)
          Ay.append(n.indice)
          Av.append(-1./n.dx)
          #
          Ax.append(n.indice)
          Ay.append(westElements[0].indice)
          Av.append(1./n.dx)
        # Le noeud à gauche est 2x plus grand:
        else:
          # print("dx i="+str(n.indice))
          # le noeud est au nord
          if n.y <= westElements[0].y:
            # print("N")
            Ax.append(n.indice)
            Ay.append(n.indice)
            Av.append(-1./(3*n.dx))
            #
            Ax.append(n.indice)
            Ay.append(southElements[0].indice)
            Av.append(-1./(3*n.dx))
              #
            Ax.append(n.indice)
            Ay.append(westElements[0].indice)
            Av.append(2./(3*n.dx))
          # le noeud est au sud
          else:
            # print("S")
            Ax.append(n.indice)
            Ay.append(n.indice)
            Av.append(-1./(3*n.dx))
            #
            Ax.append(n.indice)
            Ay.append(northElements[0].indice)
            Av.append(-1./(3*n.dx))
              #
            Ax.append(n.indice)
            Ay.append(westElements[0].indice)
            Av.append(2./(3*n.dx))
            
    
    # Dérivée en y:
    ## On est sur le bord en bas
    if len(northElements) == 0:
      if bc == "N":
        Ax.append(n.indice)
        Ay.append(dofs + n.indice)
        Av.append(0)
      else:
        Ax.append(n.indice)
        Ay.append(dofs + n.indice)
        Av.append(-1./n.dy)
    else:
      # Le noeud en haut est 2x plus petit:
      if len(northElements) == 2:
        # Dangling 1
        Ax.append(n.indice)
        Ay.append(dofs + n.indice)
        Av.append(-4./(3*n.dy))
        #
        Ax.append(n.indice)
        Ay.append(dofs + northElements[0].indice)
        Av.append(2./(3*n.dy))
        #
        Ax.append(n.indice)
        Ay.append(dofs + northElements[1].indice)
        Av.append(2./(3*n.dy))
      else:
        # Le noeud en haut est de la même taille
        if northElements[0].dx == n.dx:
          # Regular
          Ax.append(n.indice)
          Ay.append(dofs + n.indice)
          Av.append(-1./n.dy)
          #
          Ax.append(n.indice)
          Ay.append(dofs + northElements[0].indice)
          Av.append(1./n.dy)
        # Le noeud en bas est 2x plus grand:
        else:
          # print("dy i="+str(n.indice))
          # le noeud est à l'ouest
          if n.x <= northElements[0].x:
            # print("W")
            Ax.append(n.indice)
            Ay.append(dofs + n.indice)
            Av.append(-1./(3*n.dx))
            #
            Ax.append(n.indice)
            Ay.append(dofs + eastElements[0].indice)
            Av.append(-1./(3*n.dx))
              #
            Ax.append(n.indice)
            Ay.append(dofs + northElements[0].indice)
            Av.append(2./(3*n.dx))
          # le noeud est à l'est
          else:
            # print("E")
            Ax.append(n.indice)
            Ay.append(dofs + n.indice)
            Av.append(-1./(3*n.dx))
            #
            Ax.append(n.indice)
            Ay.append(dofs + westElements[0].indice)
            Av.append(-1./(3*n.dx))
              #
            Ax.append(n.indice)
            Ay.append(dofs + northElements[0].indice)
            Av.append(2./(3*n.dx))
    
  A = -sparse.csr_matrix((Av, (Ax, Ay)), shape=[dofs, 2*dofs])
  return A # cupyx.scipy.sparse.csr_matrix(A)

def gradOperator_b(view, bc = "N"):
  Av = []
  Ax = []
  Ay = []
  elements = view.getElements()
  dofs = len(elements)
  for n in elements:
    eastElements = view.getEastNeighbours(n)
    westElements = view.getWestNeighbours(n)
    northElements = view.getNorthNeighbours(n)
    southElements = view.getSouthNeighbours(n)
    # Dérivée en x:
    ## On est sur le bord droit
    if len(westElements) == 0:
      if bc == "N":
        Ax.append(n.indice)
        Ay.append(n.indice)
        Av.append(0)
      else:
        Ax.append(n.indice)
        Ay.append(n.indice)
        Av.append(-1./n.dx)
    else:
      # Le noeud à gauche est 2x plus petit:
      if len(westElements) == 2:
        # Dangling 1
        Ax.append(n.indice)
        Ay.append(n.indice)
        Av.append(-4./(3*n.dx))
        #
        Ax.append(n.indice)
        Ay.append(westElements[0].indice)
        Av.append(2./(3*n.dx))
        #
        Ax.append(n.indice)
        Ay.append(westElements[1].indice)
        Av.append(2./(3*n.dx))
      else:
        # Le noeud à gauche est de la même taille
        if westElements[0].dx == n.dx:
          # Regular
          Ax.append(n.indice)
          Ay.append(n.indice)
          Av.append(-1./n.dx)
          #
          Ax.append(n.indice)
          Ay.append(westElements[0].indice)
          Av.append(1./n.dx)
        # Le noeud à gauche est 2x plus grand:
        else:
          # print("dx i="+str(n.indice))
          # le noeud est au nord
          if n.y <= westElements[0].y:
            # print("N")
            Ax.append(n.indice)
            Ay.append(n.indice)
            Av.append(-1./(3*n.dx))
            #
            Ax.append(n.indice)
            Ay.append(southElements[0].indice)
            Av.append(-1./(3*n.dx))
              #
            Ax.append(n.indice)
            Ay.append(westElements[0].indice)
            Av.append(2./(3*n.dx))
          # le noeud est au sud
          else:
            # print("S")
            Ax.append(n.indice)
            Ay.append(n.indice)
            Av.append(-1./(3*n.dx))
            #
            Ax.append(n.indice)
            Ay.append(northElements[0].indice)
            Av.append(-1./(3*n.dx))
              #
            Ax.append(n.indice)
            Ay.append(westElements[0].indice)
            Av.append(2./(3*n.dx))
            
    
    # Dérivée en y:
    ## On est sur le bord en bas
    if len(northElements) == 0:
      if bc == "N":
        Ax.append(dofs + n.indice)
        Ay.append(n.indice)
        Av.append(0)
      else:
        Ax.append(dofs + n.indice)
        Ay.append(n.indice)
        Av.append(-1./n.dy)
    else:
      # Le noeud en haut est 2x plus petit:
      if len(northElements) == 2:
        # Dangling 1
        Ax.append(dofs + n.indice)
        Ay.append(n.indice)
        Av.append(-4./(3*n.dy))
        #
        Ax.append(dofs + n.indice)
        Ay.append(northElements[0].indice)
        Av.append(2./(3*n.dy))
        #
        Ax.append(dofs + n.indice)
        Ay.append(northElements[1].indice)
        Av.append(2./(3*n.dy))
      else:
        # Le noeud en haut est de la même taille
        if northElements[0].dx == n.dx:
          # Regular
          Ax.append(dofs + n.indice)
          Ay.append(n.indice)
          Av.append(-1./n.dy)
          #
          Ax.append(dofs + n.indice)
          Ay.append(northElements[0].indice)
          Av.append(1./n.dy)
        # Le noeud en bas est 2x plus grand:
        else:
          # print("dy i="+str(n.indice))
          # le noeud est à l'ouest
          if n.x <= northElements[0].x:
            # print("W")
            Ax.append(dofs + n.indice)
            Ay.append(n.indice)
            Av.append(-1./(3*n.dx))
            #
            Ax.append(dofs + n.indice)
            Ay.append(eastElements[0].indice)
            Av.append(-1./(3*n.dx))
              #
            Ax.append(dofs + n.indice)
            Ay.append(northElements[0].indice)
            Av.append(2./(3*n.dx))
          # le noeud est à l'est
          else:
            # print("E")
            Ax.append(dofs + n.indice)
            Ay.append(n.indice)
            Av.append(-1./(3*n.dx))
            #
            Ax.append(dofs + n.indice)
            Ay.append(westElements[0].indice)
            Av.append(-1./(3*n.dx))
              #
            Ax.append(dofs + n.indice)
            Ay.append(northElements[0].indice)
            Av.append(2./(3*n.dx))
    
  A = -sparse.csr_matrix((Av, (Ax, Ay)), shape=[2*dofs, dofs])
  return A # cupyx.scipy.sparse.csr_matrix(A)

def gradOperator_centered(mesh, bc = "N"):
    return 0.5*(gradOperator(mesh, bc) + gradOperator_b(mesh, bc))
