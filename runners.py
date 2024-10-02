import quadmesh
import projection
import numpy as np
import utils
from scipy.sparse import bmat
from scipy import sparse
import models

from PIL import Image

# Coarse-to-fine runner
class DenoisingCoarseToFineRunner(object):
    def __init__(self, algorithm, w, h, NRef):
        self.algorithm = algorithm
        self.w = w
        self.h = h
        self.NRef = NRef

    def init(self, model, g, sigma):
        self.model = model
        self.g = g
        self.sigma = sigma
        
    def adaptMesh(self, view, err, model, u):
        elements = view.getElements()
        dofs = len(elements)

        #TMP
        viewImage = quadmesh.QuadMeshLeafView(self.w, self.h, self.w, self.h)
        viewImage.create()
        viewImage.computeIndices()
        
        meshPrev = view.copy()
        viewPrev = quadmesh.QuadMeshLeafView(self.w, self.h, int(self.w/2**(self.NRef)), int(self.h/2**(self.NRef)), root=meshPrev)
        
        iSorted = np.argsort(err)[::-1]

        # Compute bulk criterion
        print("compute projections from mesh to image...")
        PInv = projection.projectionMatrix(view, viewImage)
        uProj = PInv@u
        residual = 0.5*(uProj-self.g)**2
        residualProj = projection.projectionMatrix(viewImage, view)@residual

        # Mark elements
        elementsToRefine = []
        theta_mark = 0.6
        sum_eta = np.sum(err)
        current_sum_eta = 0
        for n in range(0, dofs):
            e = elements[iSorted[n]]
            if residualProj[e.indice] > self.sigma**2/2: #######
                elementsToRefine.append(e)
                current_sum_eta = current_sum_eta + err[e.indice]
                if current_sum_eta >= theta_mark*sum_eta:
                    break

        # Refine marked elements
        for e in elementsToRefine:
            view.refine(e)
        view.balance()
        view.balance()
        view.balance()
        view.balance()
        view.balance()
        view.computeIndices()
        
        return viewPrev

    def run(self):
        viewImage = quadmesh.QuadMeshLeafView(self.w, self.h, self.w, self.h)
        viewImage.create()
        viewImage.computeIndices()
        
        view = quadmesh.QuadMeshLeafView(self.w, self.h, int(self.w/2**(self.NRef)), int(self.h/2**(self.NRef)))
        view.create()
        view.computeIndices()
        
        print("compute projections from image to mesh...")
        P = projection.projectionMatrix(viewImage, view)
        # copy model
        model = models.L1L2TVModel()
        model.alpha1  = P@self.model.alpha1
        model.alpha2  = P@self.model.alpha2
        model.lambdaa = P@self.model.lambdaa
        model.beta    = P@self.model.beta
        model.gamma1  = P@self.model.gamma1
        model.gamma2  = P@self.model.gamma2
        # copy data
        g = P@self.g
        n = 0
        while True:
            # view.show()
            elements = view.getElements()
            dofs = len(elements)
            level_max = elements[0].level
            for e in elements:
                if e.level > level_max:
                    level_max = e.level              

            # quadmesh.showQMeshFunction(view, np.ones(dofs))
            # quadmesh.showQMeshFunction(view, np.ones(dofs), pathname="results/mesh-"+str(n)+".png")
            # print([n, dofs])
            
            print("dofs: "+str(dofs)+" ("+str(100*dofs/self.w/self.h)+")")
        
            print("run algorithm...")
            self.algorithm.init(view, g, model)
            [u, p1, p2, err] = self.algorithm.run() 
            # quadmesh.showQMeshFunction(view, u, pathname="results/u-"+str(n)+".png", displayEdges=False)

            if level_max > self.NRef+1:
                break
                
            print("adapt mesh...")
            viewPrev = self.adaptMesh(view, np.abs(err), model, u)
            
            print("compute projections...")
            P = projection.projectionMatrix(viewImage, view)
            M = projection.projectionMatrix(viewPrev, view)    
            model.alpha1  = M@model.alpha1
            model.alpha2  = M@model.alpha2
            model.lambdaa = M@model.lambdaa
            model.beta    = M@model.beta
            model.gamma1  = M@model.gamma1
            model.gamma2  = M@model.gamma2
            g = P@self.g
            
            n = n + 1
        
        print("compute projections from mesh to image...")
        PInv = projection.projectionMatrix(view, viewImage)
        return PInv@u
        
    def __str__(self):
        return "Runner: coarse-to-fine runner:\n - NRef    = "+str(self.NRef)

class SimpleRunner(object):
    def __init__(self, algorithm, w, h):
        self.algorithm = algorithm
        self.w = w
        self.h = h

    def init(self, model, f0, f1):
        self.model = model
        self.f0 = f0
        self.f1 = f1
        
        self.view = quadmesh.QuadMeshLeafView(self.w, self.h, self.w, self.h)
        self.view.create()
        self.view.computeIndices()
        dofs = len(self.view.getElements())
        
        self.algorithm.init(self.view, self.f0, self.f1, np.zeros(2*dofs), self.model)

    def run(self):
        print("run algorithm...")
        [u, p1, p2, err] = self.algorithm.run()        
        return [u, p1, p2]

    def __str__(self):
        return "Runner: simple runner"

class AllInOneRunner(object):
    def __init__(self, algorithm, w, h, NRef, ctf_epsilon, ctf_mark):
        self.algorithm = algorithm
        self.w = w
        self.h = h
        self.NRef = NRef
        self.ctf_epsilon = ctf_epsilon
        self.ctf_mark = ctf_mark

    def init(self, model, f0, f1):
        self.model = model
        self.f0 = f0
        self.f1 = f1

    def warp_error(self, fwk, fwk_prev, f0):
        l2_fwk_f0      = np.sqrt( np.sum( (fwk - f0)**2 ) )
        l2_fwk_prev_f0 = np.sqrt( np.sum( (fwk_prev - f0)**2 ) )
        return (l2_fwk_prev_f0-l2_fwk_f0)/l2_fwk_prev_f0
        
    def adaptMesh(self, view, err, model):
        elements = view.getElements()
        dofs = len(elements)
        
        meshPrev = view.copy()
        viewPrev = quadmesh.QuadMeshLeafView(self.w, self.h, int(self.w/2**self.NRef), int(self.h/2**self.NRef), root=meshPrev)
        
        iSorted = np.argsort(err)[::-1]
        
        toRefine = max( int(self.ctf_mark*dofs), 1)

        # Mark elements
        elementsToRefine = []
        refined = 0
        n = 0
        while refined < toRefine and n < dofs:
            e = elements[iSorted[n]]
            wwPix = e.dx*e.dy
            elementsToRefine.append(e)
            refined = refined + 1
            n = n + 1

        # Refine marked elements
        for e in elementsToRefine:
            view.refine(e)
        view.balance()
        view.balance()
        view.balance()
        view.balance()
        view.balance()
        view.balance()
        view.computeIndices()
        
        return viewPrev

    def run(self):
        h = self.h
        w = self.w

        fw = self.f1
        fw_prev = 0
        
        viewImage = quadmesh.QuadMeshLeafView(self.w, self.h, self.w, self.h)
        viewImage.create()
        viewImage.computeIndices()

        view = quadmesh.QuadMeshLeafView(self.w, self.h, int(w/2**self.NRef), int(h/2**self.NRef))
        view.create()
        view.computeIndices()

        elements = view.getElements()
        dofs = len(elements)
        
        print("compute projections from image to mesh...")
        P = projection.projectionMatrix(viewImage, view)
        PInv = projection.projectionMatrix(view, viewImage)
        PInv2N = bmat([ [ PInv, sparse.csr_matrix((w*h, dofs)) ], [ sparse.csr_matrix((w*h, dofs)), PInv ] ])
        # copy model
        model = models.L1L2TVModel()
        model.alpha1  = P@self.model.alpha1
        model.alpha2  = P@self.model.alpha2
        model.lambdaa = P@self.model.lambdaa
        model.beta    = P@self.model.beta
        model.gamma1  = P@self.model.gamma1
        model.gamma2  = P@self.model.gamma2
        # copy data
        f0_proj = P@self.f0
        fw_proj = P@fw
        
        uk = 0
        NRef = 0
        viewPrev = 0        
        while True:
            print("dofs: "+str(dofs)+" ("+str(100*dofs/w/h)+"%)")

            # quadmesh.showQMeshFunction(view, np.ones(dofs), pathname="mesh-"+str(NRef)+".png")

            print("run algorithm...")
            self.algorithm.init(view, f0_proj, fw_proj, np.zeros(2*dofs), model)
            [u, p1, p2, err] = self.algorithm.run() 

            print("warp image...")
            # PInv = projection.projectionMatrix(view, viewImage)
            # PInv2N = bmat([ [ PInv, sparse.csr_matrix((w*h, dofs)) ], [ sparse.csr_matrix((w*h, dofs)), PInv ] ])
            uk = uk + PInv2N@u
            fw_prev = fw
            fw = utils.apply(self.f1, uk[0:self.w*self.h], uk[self.w*self.h:2*self.w*self.h], self.w, self.h)
            # TODO : normalize flag
            fw = fw/(np.sum(fw)/(self.w*self.h))

            # stopping criterion
            crit = self.warp_error(fw, fw_prev, self.f0)
            print("warp criterion: "+str(crit))
            if crit < self.ctf_epsilon:
                
                NRef = NRef + 1
                if NRef > self.NRef:
                    break
                
                print("adapt mesh ("+str(NRef)+"/"+str(self.NRef)+")...")
                # quadmesh.showQMeshFunction(view, np.abs(err))
                viewPrev = self.adaptMesh(view, np.abs(err), model)      
                elements = view.getElements()
                dofs = len(elements)        
    
                print("compute projections...")
                P = projection.projectionMatrix(viewImage, view)
                PInv = projection.projectionMatrix(view, viewImage)
                PInv2N = bmat([ [ PInv, sparse.csr_matrix((w*h, dofs)) ], [ sparse.csr_matrix((w*h, dofs)), PInv ] ])
                M = projection.projectionMatrix(viewPrev, view)    
                model.alpha1  = M@model.alpha1
                model.alpha2  = M@model.alpha2
                model.lambdaa = M@model.lambdaa
                model.beta    = M@model.beta
                model.gamma1  = M@model.gamma1
                model.gamma2  = M@model.gamma2

            print("project data...")
            f0_proj = P@self.f0
            fw_proj = P@fw

        return [uk, p1, p2]

    def __str__(self):
        return "Runner: all-in-one runner:\n - NRef    = "+str(self.NRef)+"\n - epsilon = "+str(self.ctf_epsilon)