import numpy as np
from scipy import sparse
from alive_progress import alive_bar

def segmentIntersection(x1, x2, xtilde1, xtilde2):
  # assume : 0 <= x1 <= x2 and 0 <= xtilde1 <= xtilde2
  if x1 <= x2 and xtilde1 <= xtilde2:
    if x1 > xtilde1:
      # invert [x1,x2] and [xtilde1,xtilde2]
      tmp = x1
      x1 = xtilde1
      xtilde1 = tmp
      tmp = x2
      x2 = xtilde2
      xtilde2 = tmp
    # from here : 0 <= x1 <= xtilde1
    if x1 <= xtilde1:
      if x2 < xtilde1: # 0 <= x1 <= x2 <= xtilde1 <= xtilde2
        return [-1, -1] # emptyset
      elif x2 == xtilde1:
        return [x2, x2] # singleton
      elif xtilde1 <= x2 and xtilde2 <= x2: # 0 <= x1 <= xtilde1 <= xtilde2 <= x2
        return [xtilde1, xtilde2]
      elif xtilde1 <= x2 and x2 <= xtilde2: # 0 <= x1 <= xtilde1 <= x2 <= xtilde2
        return [xtilde1, x2]
  print('error segmentIntersection')

def quadIntersection(quad1, quad2):
  [x1, x2, y1, y2] = quad1
  [xtilde1, xtilde2, ytilde1, ytilde2] = quad2
  [ix1, ix2] = segmentIntersection(x1, x2, xtilde1, xtilde2)
  [iy1, iy2] = segmentIntersection(y1, y2, ytilde1, ytilde2)
  return [ix1, ix2, iy1, iy2]

def quadMeasure(quad):
  [x1, x2, y1, y2] = quad
  return np.abs((x2-x1)*(y2-y1))

def computeBasisFunctionSupport(n):
  return [n.x-n.dx/2, n.x+n.dx/2, n.y-n.dy/2, n.y+n.dy/2]

# def projectionMatrix(viewSrc, viewDst): 
#   nodesSrc = viewSrc.getElements()
#   nodesDst = viewDst.getElements()
#   Mx = []
#   My = []
#   Mv = []
#   for nDst in nodesDst:
#     eDst = computeBasisFunctionSupport(nDst)
#     nSrcInter = viewSrc.findElementsInRectangle(nDst.x-nDst.dx/2, nDst.y-nDst.dy/2, nDst.x+nDst.dx/2, nDst.y+nDst.dy/2) # can be faster by selecting less elements
#     for nSrc in nSrcInter:
#       eSrc = computeBasisFunctionSupport(nSrc)
#       intersection = quadIntersection(eDst, eSrc)
#       if quadMeasure(intersection) > 0:
#         Mx.append(nDst.indice)
#         My.append(nSrc.indice)
#         Mv.append(quadMeasure(intersection)/quadMeasure(eDst))
#   M = sparse.csr_matrix((Mv, (Mx, My)), shape=[len(nodesDst), len(nodesSrc)])
#   return M

def projectionMatrix(viewSrc, viewDst): 
    nodesSrc = viewSrc.getElements()
    nodesDst = viewDst.getElements()
    
    Mx, My, Mv = [], [], []
    
    # Cache basis function supports for both source and destination
    basisDst = [computeBasisFunctionSupport(nDst) for nDst in nodesDst]
    basisSrc = [computeBasisFunctionSupport(nSrc) for nSrc in nodesSrc]

    # To optimize the selection of nodes within a region, we can compute a spatial index.
    # However, in this example, we'll keep the existing logic, but be mindful of optimization here.

    for i, nDst in enumerate(nodesDst):
        eDst = basisDst[i]
        
        # Find elements in the rectangle region
        nSrcInter = viewSrc.findElementsInRectangle(
            nDst.x - nDst.dx / 2,
            nDst.y - nDst.dy / 2,
            nDst.x + nDst.dx / 2,
            nDst.y + nDst.dy / 2
        )
        
        for nSrc in nSrcInter:
            eSrc = basisSrc[nSrc.indice]
            intersection = quadIntersection(eDst, eSrc)
            quadMeasure_intersection = quadMeasure(intersection)
            if quadMeasure_intersection > 0:
                Mx.append(nDst.indice)
                My.append(nSrc.indice)
                Mv.append(quadMeasure_intersection / quadMeasure(eDst))

    # Construct sparse matrix
    M = sparse.csr_matrix((Mv, (Mx, My)), shape=(len(nodesDst), len(nodesSrc)))
    return M