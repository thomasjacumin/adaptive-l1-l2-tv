import numpy as np
import utils
from scipy.sparse import bmat
from scipy import sparse

class Warping(object):
    def __init__(self, runner, f0, f1, w, h, epsilon, max_it, normalize):
        self.runner = runner
        self.f0 = f0
        self.f1 = f1
        self.w = w
        self.h = h
        self.epsilon = epsilon
        self.max_it = max_it
        self.normalize = normalize

    def init(self, model):
        self.model = model
        
    def warp_error(self, fwk, fwk_prev, f0):
        l2_fwk_f0      = np.sqrt( np.sum( (fwk - f0)**2 ) )
        l2_fwk_prev_f0 = np.sqrt( np.sum( (fwk_prev - f0)**2 ) )
        # qmesh.showQMeshFunction(qmesh.QMesh(self.w,self.h), fwk_prev, displayEdges=False)
        # print(l2_fwk_prev_f0)
        # print(l2_fwk_f0)
        return (l2_fwk_prev_f0-l2_fwk_f0)/l2_fwk_prev_f0

    def run(self):
        fw = self.f1
        fw_prev = 0
        uk = 0
        for i in range(0,self.max_it):
            print("run runner...")
            self.runner.init(self.model, self.f0, fw)
            [u, p1, p2] = self.runner.run()
            uk = uk + u
            # print("saving flo file...")
            # utils.saveFlo(self.w, self.h, uk[0:self.w*self.h], uk[self.w*self.h:2*self.w*self.h], "results/convergence/"+str(i)+".flo")
            print("warp image...")
            fw_prev = fw
            fw = utils.apply(self.f1, uk[0:self.w*self.h], uk[self.w*self.h:2*self.w*self.h], self.w, self.h)
            # qmesh.showQMeshFunction(qmesh.QMesh(self.w,self.h), u[0:self.w*self.h])
            # qmesh.showQMeshFunction(qmesh.QMesh(self.w,self.h), u[self.w*self.h:2*self.w*self.h])
            # fw = utils.apply_backward_optical_flow(self.f1.reshape([self.h, self.w]), (u[0:self.w*self.h]).reshape([self.h, self.w]), (u[self.w*self.h:2*self.w*self.h]).reshape([self.h, self.w])).flatten()
            if self.normalize:
                fw = fw/(np.sum(fw)/(self.w*self.h))
            # utils.saveImage(self.w, self.h, fw, "results/convergence/fw-"+str(i)+".png")
            # qmesh.showQMeshFunction(qmesh.QMesh(self.w,self.h), fw)
            crit = self.warp_error(fw, fw_prev, self.f0)
            print("warp stopping criterion: "+str(crit))
            if crit < self.epsilon:
                break
        return [uk, p1, p2]

    def __str__(self):
        return "Warping:\n - epsilon   = "+str(self.epsilon)+"\n - max_it    = "+str(self.max_it)+"\n - normalize = "+str(self.normalize)