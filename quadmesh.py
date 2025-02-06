import numpy as np
import matplotlib.pyplot as plt

NORTH_WEST = 0
NORTH_EAST = 1
SOUTH_EAST = 2
SOUTH_WEST = 3

class QuadMesh(object):
    def __init__(self, parent=None, childNumber=None, x=None, y=None, dx=None, dy=None, level=None):
        self.parent = parent
        self.childNumber = childNumber
        
        self.children = []
        
        self.x = x
        self.y = y
        self.indice = None
        self.dx = dx
        self.dy = dy
        self.level = level

    def __str__(self):
        return str(self.x)+" "+str(self.y)+" "+str(self.dx)+" "+str(self.dy)+" "+str(self.indice)

class QuadMeshLeafView(object):
    def __init__(self, w, h, nx, ny, root=None):
        if root == None:
            self.root = QuadMesh()
            self.root.x = w/2
            self.root.y = h/2
            self.root.dx = w
            self.root.dy = h
            self.root.level = 0
        else:
            self.root = root
        self.nx = nx
        self.ny = ny
        self.w = w
        self.h = h
        # self.elements = []
        # self.dofs = None
        # self.northNeighbours = {}
        # self.eastNeighbours = {}
        # self.southNeighbours = {}
        # self.westNeighbours = {}

    def create(self):
        self.root.children = []
        dx = self.root.dx/self.nx
        dy = self.root.dy/self.ny
        for i in range(0, self.ny):
            for j in range(0, self.nx):
                self.root.children.append(QuadMesh(parent=self.root, childNumber=j+self.nx*i, x=dx*j+dx/2, y=dy*i+dy/2, dx=dx, dy=dy, level=1))
        # self.elements = []
        # self.dofs = None
        # self.northNeighbours = {}
        # self.eastNeighbours = {}
        # self.southNeighbours = {}
        # self.westNeighbours = {}

    def computeIndices(self):
        elements = self.getElements()
        for i in range(0, len(elements) ):
            elements[i].indice = i
            
    def _getElements(self, node, elements):
        if len(node.children) > 0:
            for child in node.children:
                self._getElements(child, elements)
        else:
            elements.append(node)
            
    def getElements(self):
        # if len( self.elements ) == 0:
        #     self._getElements(self.root, self.elements)
        # return self.elements
        elements = []
        self._getElements(self.root, elements)
        return elements

    # def getDofs(self):
    #     # if self.dofs == None:
    #     #     self.dofs = len(self.getElements())
    #     # return self.dofs
    #     return len(self.getElements())

    def getElementByIndice(self, indice):
        elements = self.getElements()
        return elements[indice]

    def refine(self, node):
        node.children = []
        node.children.append(QuadMesh(parent=node, childNumber=NORTH_WEST, x=node.x-node.dx/4, y=node.y-node.dy/4, dx=node.dx/2, dy=node.dy/2, level=node.level+1))
        node.children.append(QuadMesh(parent=node, childNumber=NORTH_EAST, x=node.x+node.dx/4, y=node.y-node.dy/4, dx=node.dx/2, dy=node.dy/2, level=node.level+1))
        node.children.append(QuadMesh(parent=node, childNumber=SOUTH_EAST, x=node.x+node.dx/4, y=node.y+node.dy/4, dx=node.dx/2, dy=node.dy/2, level=node.level+1))
        node.children.append(QuadMesh(parent=node, childNumber=SOUTH_WEST, x=node.x-node.dx/4, y=node.y+node.dy/4, dx=node.dx/2, dy=node.dy/2, level=node.level+1))
        # self.elements = []

    def refineByIndice(self, indice):
        self.refine( self.getElementByIndice(indice) )

    def _getSouthElements(self, node, elements):
        if len(node.children) > 0:
            self._getSouthElements(node.children[SOUTH_EAST], elements)
            self._getSouthElements(node.children[SOUTH_WEST], elements)
        else:
            elements.append(node)

    def _getNorthElements(self, node, elements):
        if len(node.children) > 0:
            self._getNorthElements(node.children[NORTH_EAST], elements)
            self._getNorthElements(node.children[NORTH_WEST], elements)
        else:
            elements.append(node)

    def _getEastElements(self, node, elements):
        if len(node.children) > 0:
            self._getEastElements(node.children[NORTH_EAST], elements)
            self._getEastElements(node.children[SOUTH_EAST], elements)
        else:
            elements.append(node)

    def _getWestElements(self, node, elements):
        if len(node.children) > 0:
            self._getWestElements(node.children[NORTH_WEST], elements)
            self._getWestElements(node.children[SOUTH_WEST], elements)
        else:
            elements.append(node)

    def _getNorthNeighbours(self, node):
        if node == self.root: # if root node
            return None
        if node.parent == self.root: # if it is an element of the uniform part
            iChildren = node.childNumber
            if iChildren < self.nx: # top element
                return None
            else:
                return self.root.children[iChildren-self.nx]
        else: # it is a real quadMesh now
            if node == node.parent.children[SOUTH_WEST]:
                return node.parent.children[NORTH_WEST]
            if node == node.parent.children[SOUTH_EAST]:
                return node.parent.children[NORTH_EAST]
            mu = self._getNorthNeighbours(node.parent)
            if mu == None or len(mu.children) == 0:
                return mu
            else:
                if node == node.parent.children[NORTH_WEST]:
                    return mu.children[SOUTH_WEST]
                else:
                    return mu.children[SOUTH_EAST]

    def _getSouthNeighbours(self, node):
        if node == self.root: # if root node
            return None
        if node.parent == self.root: # if it is an element of the uniform part
            iChildren = node.childNumber
            if iChildren+self.nx >= self.nx*self.ny: # bottom element
                return None
            else:
                return self.root.children[iChildren+self.nx]
        else: # it is a real quadMesh now
            if node == node.parent.children[NORTH_WEST]:
                return node.parent.children[SOUTH_WEST]
            if node == node.parent.children[NORTH_EAST]:
                return node.parent.children[SOUTH_EAST]
            mu = self._getSouthNeighbours(node.parent)
            if mu == None or len(mu.children) == 0:
                return mu
            else:
                if node == node.parent.children[SOUTH_WEST]:
                    return mu.children[NORTH_WEST]
                else:
                    return mu.children[NORTH_EAST]

    def _getEastNeighbours(self, node):
        if node == self.root: # if root node
            return None
        if node.parent == self.root: # if it is an element of the uniform part
            iChildren = node.childNumber
            if iChildren%self.nx == self.nx-1: # east element
                return None
            else:
                return self.root.children[iChildren+1]
        else: # it is a real quadMesh now
            if node == node.parent.children[NORTH_WEST]:
                return node.parent.children[NORTH_EAST]
            if node == node.parent.children[SOUTH_WEST]:
                return node.parent.children[SOUTH_EAST]
            mu = self._getEastNeighbours(node.parent)
            if mu == None or len(mu.children) == 0:
                return mu
            else:
                if node == node.parent.children[SOUTH_EAST]:
                    return mu.children[SOUTH_WEST]
                else:
                    return mu.children[NORTH_WEST]

    def _getWestNeighbours(self, node):
        if node == self.root: # if root node
            return None
        if node.parent == self.root: # if it is an element of the uniform part
            iChildren = node.childNumber
            if iChildren%self.nx == 0: # west element
                return None
            else:
                return self.root.children[iChildren-1]
        else: # it is a real quadMesh now
            if node == node.parent.children[NORTH_EAST]:
                return node.parent.children[NORTH_WEST]
            if node == node.parent.children[SOUTH_EAST]:
                return node.parent.children[SOUTH_WEST]
            mu = self._getWestNeighbours(node.parent)
            if mu == None or len(mu.children) == 0:
                return mu
            else:
                if node == node.parent.children[SOUTH_WEST]:
                    return mu.children[SOUTH_EAST]
                else:
                    return mu.children[NORTH_EAST]

    def getNorthNeighbours(self, node):
        # if node.indice in self.northNeighbours:
        #     return self.northNeighbours[node.indice]
        # else:
        north = self._getNorthNeighbours(node)
        elements = []
        if north != None:
            self._getSouthElements(north, elements)
        return elements
            # self.northNeighbours[node.indice] = elements
            # return self.northNeighbours[node.indice]

    def getSouthNeighbours(self, node):
        # if node.indice in self.southNeighbours:
        #     return self.southNeighbours[node.indice]
        # else:
        south = self._getSouthNeighbours(node)
        elements = []
        if south != None:
            self._getNorthElements(south, elements)
        return elements
            # self.southNeighbours[node.indice] = elements
            # return self.southNeighbours[node.indice]

    def getEastNeighbours(self, node):
        # if node.indice in self.eastNeighbours:
        #     return self.eastNeighbours[node.indice]
        # else:
        east = self._getEastNeighbours(node)
        elements = []
        if east != None:
            self._getWestElements(east, elements)
        return elements
            # self.eastNeighbours[node.indice] = elements
            # return self.eastNeighbours[node.indice]

    def getWestNeighbours(self, node):
        # if node.indice in self.westNeighbours:
        #     return self.westNeighbours[node.indice]
        # else:
        west = self._getWestNeighbours(node)
        elements = []
        if west != None:
            self._getEastElements(west, elements)
        return elements
            # self.westNeighbours[node.indice] = elements
            # return self.westNeighbours[node.indice]

    def balance(self):
        elementsToCheck = self.getElements()
        while len(elementsToCheck) > 0:
            e = elementsToCheck.pop()
            nn = self.getNorthNeighbours(e)
            ne = self.getEastNeighbours(e)
            ns = self.getSouthNeighbours(e)
            nw = self.getWestNeighbours(e)
            if len(nn)>2 or len(ne)>2 or len(ns)>2 or len(nw)>2:
                self.refine(e)
                for c in e.children:
                    elementsToCheck.append(c)

    def show(self):
        elements = self.getElements()
        fig, ax = plt.subplots(figsize=(30,30))
        X = [e.x for e in elements]
        Y = [e.y for e in elements]
        padding = 1
        ax.set_ylim( self.root.dy + padding, 0 - padding)
        ax.set_xlim( 0 - padding, self.root.dx + padding)
          
        x = []
        y = []
        for e in elements:
            x.append(e.x)
            y.append(e.y)
        ax.scatter(x,y, color='b')
    
        for e in elements:
            dx = e.dx
            dy = e.dy
            ax.plot([e.x-dx/2,e.x+dx/2], [e.y-dy/2,e.y-dy/2], 'b-')
            ax.plot([e.x-dx/2,e.x+dx/2], [e.y+dy/2,e.y+dy/2], 'b-')
            ax.plot([e.x-dx/2,e.x-dx/2], [e.y-dy/2,e.y+dy/2], 'b-')
            ax.plot([e.x+dx/2,e.x+dx/2], [e.y-dy/2,e.y+dy/2], 'b-')
            
        opt = {'head_width': 0.01, 'head_length': 0.004, 'width': 0.001, 'length_includes_head': True, 'color': 'r'}
        eps = 0.001
        for e in elements:
            for ee in self.getNorthNeighbours(e):
                ax.arrow(e.x,e.y-eps, ee.x-e.x,ee.y-e.y+eps, **opt)
            for ee in self.getEastNeighbours(e):
                ax.arrow(e.x,e.y-eps, ee.x-e.x,ee.y-e.y+eps, **opt)
            for ee in self.getSouthNeighbours(e):
                ax.arrow(e.x,e.y-eps, ee.x-e.x,ee.y-e.y+eps, **opt)
            for ee in self.getWestNeighbours(e):
                ax.arrow(e.x,e.y-eps, ee.x-e.x,ee.y-e.y+eps, **opt)
        
        for e in elements:
            ax.text(e.x+eps, e.y+eps, str(e.indice))  
    
        plt.tight_layout()
        plt.show()

    def _copy(self, nodeSrc, nodeDst):
        # nodeDst.parent = nodeSrc.parent
        nodeDst.childNumber = nodeSrc.childNumber
        nodeDst.x = nodeSrc.x
        nodeDst.y = nodeSrc.y
        nodeDst.indice = nodeSrc.indice
        nodeDst.dx = nodeSrc.dx
        nodeDst.dy = nodeSrc.dy
        nodeDst.level = nodeSrc.level
        if len(nodeSrc.children) > 0:
            for childSrc in nodeSrc.children: 
                childDst = QuadMesh(parent=nodeDst)
                nodeDst.children.append(childDst)
                self._copy(childSrc, childDst)

    def copy(self):
        rootDst = QuadMesh()
        self._copy(self.root, rootDst)
        return rootDst

    def _findElement(self, x, y, node):
        if len(node.children) > 0:
            xRel = x-node.x+node.dx/2
            yRel = y-node.y+node.dy/2
            if xRel < node.dx/2 and yRel < node.dy/2:
                return self._findElement(x, y, node.children[NORTH_WEST])
            if xRel >= node.dx/2 and yRel < node.dy/2:
                return self._findElement(x, y, node.children[NORTH_EAST])
            if xRel < node.dx/2 and yRel >= node.dy/2:
                return self._findElement(x, y, node.children[SOUTH_WEST])
            if xRel >= node.dx/2 and yRel >= node.dy/2:
                return self._findElement(x, y, node.children[SOUTH_EAST])
        else:
            return node
    
    def findElement(self, x, y):
        dx = self.root.dx/self.nx
        dy = self.root.dy/self.ny
        j = int(y/dy)
        i = int(x/dx)
        n = j*self.nx+i
        if n < self.nx*self.ny:
            element = self._findElement(x, y, self.root.children[n])
        else:
            element = None
        return element

    # def _findElementsInRectangle(self, xTL, yTL, xBR, yBR, node, elements):
    #     if len(node.children) > 0:
    #         for child in node.children:
    #             self._findElementsInRectangle(xTL, yTL, xBR, yBR, child, elements)
    #     else:
    #         elements.append(node)
        
    # def findElementsInRectangle(self, xTL, yTL, xBR, yBR):
    #     dx = self.root.dx/self.nx
    #     dy = self.root.dy/self.ny
    #     elements = []
    #     for j in range(int(yTL/dy), int(yBR/dy)+1):
    #         for i in range(int(xTL/dx), int(xBR/dx)+1):
    #             n = j*self.nx+i
    #             if n < self.nx*self.ny:
    #                self._findElementsInRectangle(xTL, yTL, xBR, yBR, self.root.children[n], elements)
    #     return elements

    def findElementsInRectangle(self, xTL, yTL, xBR, yBR):
        # Precompute dx and dy, as they do not change
        dx = self.root.dx / self.nx
        dy = self.root.dy / self.ny
        
        # Precompute integer bounds for x and y axis
        start_j = int(yTL / dy)
        end_j = int(yBR / dy)
        start_i = int(xTL / dx)
        end_i = int(xBR / dx)
        
        elements = []
        
        # Stack for iterative depth-first traversal of the tree
        stack = []
    
        # Iterate over grid points
        for j in range(start_j, end_j + 1):
            for i in range(start_i, end_i + 1):
                n = j * self.nx + i
                if n < self.nx * self.ny:
                    stack.append(self.root.children[n])
        
        # Now process the stack iteratively
        while stack:
            node = stack.pop()
            
            if len(node.children) > 0:
                for child in node.children:
                    stack.append(child)
            else:
                elements.append(node)
        
        return elements

from matplotlib.patches import Rectangle

def showQMeshFunction(view, f, pathname=None, title=None, displayEdges=True, vmin=None, vmax=None):
    elements = view.getElements()
    X = [e.x for e in elements]
    Y = [e.y for e in elements]
    aspect = np.max(X)/np.max(Y)
    fig, ax = plt.subplots(figsize=(10*aspect,10))
    padding = 1
    ax.set_ylim( view.root.dy + padding, 0 - padding)
    ax.set_xlim( 0 - padding, view.root.dx + padding)

    if vmin == None:
        cMin = np.min(f)
    else:
        cMin = vmin

    if vmax == None:
        cMax = np.max(f)
    else:
        cMax = vmax

    if title:
        ax.text(0, -0.01, title)
    # ax.text(2, 0, "min : "+str(cMin)+"    max : "+str(cMax))

    eps = 0.001
    for e in elements:
        dx = e.dx
        dy = e.dy
        if cMax-cMin > 0:
            c = (f[e.indice]-cMin)/(cMax-cMin)
        elif cMax > 1:
            c = f[e.indice]/cMax
        else:
            c = f[e.indice]
        c = np.clip(c, 0, 1)
        if displayEdges:
            ax.add_patch(Rectangle((e.x-dx/2, e.y-dy/2), dx, dy, edgecolor = 'blue', facecolor = (c,c,c), fill=True))
        else:
            ax.add_patch(Rectangle((e.x-dx/2, e.y-dy/2), dx, dy, facecolor = (c,c,c), fill=True)) 
    
    plt.tight_layout()
    if pathname != None:
        plt.savefig(pathname)
    plt.show()