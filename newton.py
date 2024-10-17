from scipy.sparse import bmat
from scipy import sparse
import numpy as np
from alive_progress import alive_bar
import operators
import error_indicators
import models
    
class L1L2TVNewtonDenoising(object):
    def __init__(self, max_it, epsilon):
        self.max_it = max_it
        self.epsilon = epsilon
        self.model = models.L1L2TVModel()

    def init(self, mesh, g, model):
        dofs = len(mesh.getElements())
        self.mesh = mesh
        self.model = model
        self.gradOp = operators.gradOperator(mesh, bc='N')
        self.divOp = operators.divOperator(mesh, bc='D')

        self.T = sparse.identity(dofs)
        self.TAdj = self.T.transpose()
        self.TAdjT = self.TAdj@self.T
        self.S = sparse.identity(dofs)
        self.SAdj =  self.S.transpose()
        self.SAdjS = self.SAdj@self.S

        self.B = sparse.diags(model.alpha2)@self.TAdjT + sparse.diags(model.beta)@self.SAdjS
        self.g = g

    def residual(self, u, p1, p2):
        mesh = self.mesh
        elements = mesh.getElements()
        dofs = len(elements)
        
        alpha1  = self.model.alpha1
        alpha2  = self.model.alpha2
        lambdaa = self.model.lambdaa
        beta    = self.model.beta
        gamma1  = self.model.gamma1
        gamma2  = self.model.gamma2
        gradOp = self.gradOp
        divOp  = self.divOp
        T = self.T
        B = self.B
        S = self.S
        g = self.g
        TAdj = self.TAdj

        rhoN = lambda mesh, v : np.abs( v[0:dofs] )
        rho2N = lambda mesh, v : np.sqrt( v[0:dofs]**2 + v[dofs:2*dofs]**2 )
        DN = lambda v: sparse.diags(v)
        D2N = lambda v: sparse.diags( np.hstack( ( v,v ) ) )
        N = lambda mesh, v : bmat([ [sparse.diags(v[0:dofs]), sparse.diags(v[dofs:2*dofs])], [sparse.diags(v[0:dofs]), sparse.diags(v[dofs:2*dofs])] ])

        mK = np.zeros(dofs)
        for e in elements:
            mK[e.indice] = e.dx*e.dy

        m1 = np.maximum( gamma1, rhoN(mesh, T@u-g) )
        m2 = np.maximum( gamma2, rho2N(mesh, gradOp@u) )

        if alpha1.all() == 0:
            gamma1 = np.zeros(dofs)
        chi1 = (rhoN(mesh, T@u-g) >= gamma1).astype(int) # gamma1 = 0 if alpha1=0
        chi2 = (rho2N(mesh, gradOp@u) >= gamma2).astype(int)
            
        residual_part_1 = TAdj@p1 - divOp@p2 - DN(alpha2)@TAdj@g + B@u
        residual_part_2 = DN(m1)@p1 - DN(alpha1)@(T@u - g)
        residual_part_3 = D2N(m2)@p2 - D2N(lambdaa)@gradOp@u
        residual = residual_part_1[0:dofs] + residual_part_2 + residual_part_3[0:dofs] + residual_part_3[dofs:2*dofs]
        residual_l2 = np.sqrt(np.sum(mK*residual**2))

        return residual_l2

    def run(self): 
        mesh = self.mesh
        elements = mesh.getElements()
        dofs = len(elements)
        
        alpha1  = self.model.alpha1
        alpha2  = self.model.alpha2
        lambdaa = self.model.lambdaa
        beta    = self.model.beta
        gamma1  = self.model.gamma1
        gamma2  = self.model.gamma2
        gradOp = self.gradOp
        divOp  = self.divOp
        T = self.T
        B = self.B
        S = self.S
        g = self.g
        TAdj = self.TAdj        
        
        rhoN = lambda mesh, v : np.abs( v[0:dofs] )
        rho2N = lambda mesh, v : np.sqrt( v[0:dofs]**2 + v[dofs:2*dofs]**2 )
        DN = lambda v: sparse.diags(v)
        D2N = lambda v: sparse.diags( np.hstack( ( v,v ) ) )
        N = lambda mesh, v : bmat([ [sparse.diags(v[0:dofs]), sparse.diags(v[dofs:2*dofs])], [sparse.diags(v[0:dofs]), sparse.diags(v[dofs:2*dofs])] ])
        
        # Loop Init
        u  = TAdj@g
        p1 = T@u
        p2 = gradOp@u
        # Loop
        with alive_bar(self.max_it, force_tty=True) as bar:
            for i in range(self.max_it):
                m1 = np.maximum( gamma1, rhoN(mesh, T@u-g) )
                m2 = np.maximum( gamma2, rho2N(mesh, gradOp@u) )
                
                chi1 = (rhoN(mesh, T@u-g) >= gamma1).astype(int) # gamma1 = 0 if alpha1=0
                chi2 = (rho2N(mesh, gradOp@u) >= gamma2).astype(int)
    
                if alpha1.all() != 0:
                    p1 = alpha1/np.maximum(alpha1, np.abs(p1))*p1
                p2 = np.hstack( (lambdaa,lambdaa))/np.maximum(np.hstack( (lambdaa,lambdaa)), np.hstack( (rho2N(mesh, p2),rho2N(mesh, p2))))*p2
            
                # gradUl = gradOp@u
                # xHl = []
                # yHl = []
                # vHl = []
                # for e in elements:
                #     if chi2[e.indice] == 1: 
                #         al = lambdaa[e.indice] - 1./m2[e.indice]*p2[e.indice]*gradUl[e.indice]
                #         bl =                   - 1./m2[e.indice]*p2[e.indice]*gradUl[e.indice + dofs]
                #         cl =                   - 1./m2[e.indice]*p2[e.indice + dofs]*gradUl[e.indice]
                #         dl = lambdaa[e.indice] - 1./m2[e.indice]*p2[e.indice + dofs]*gradUl[e.indice + dofs]
                #     else:
                #         al = lambdaa[e.indice]
                #         bl = 0
                #         cl = 0
                #         dl = lambdaa[e.indice]

                #     c = np.array([[al, bl], [cl, dl]])
                #     eigenvalues, eigenvectors = np.linalg.eigh(0.5*(c+c.transpose()))
                #     cHat = eigenvectors @ np.diag(np.maximum(eigenvalues, 0)) @ np.linalg.inv(eigenvectors) # projection to positive definite matrices space.

                #     xHl.append(e.indice)
                #     yHl.append(e.indice)
                #     vHl.append(cHat[0,0])
                #     xHl.append(e.indice + dofs)
                #     yHl.append(e.indice)
                #     vHl.append(cHat[0,1])
                #     xHl.append(e.indice)
                #     yHl.append(e.indice + dofs)
                #     vHl.append(cHat[1,0])
                #     xHl.append(e.indice + dofs)
                #     yHl.append(e.indice + dofs)
                #     vHl.append(cHat[1,1])           
                # Cl = sparse.csr_matrix((vHl, (yHl, xHl)), shape=[2*dofs, 2*dofs])
                Cl = D2N(lambdaa) - D2N(chi2)@DN(p2)@D2N(1./m2)@N(mesh, gradOp@u)
                H = B - divOp@D2N(1./m2)@Cl@gradOp + TAdj@DN(1./m1)@( DN(alpha1) - DN(chi1)@DN(1./m1)@DN(T@u-g)@DN(p1) )@T
                F = DN(alpha2)@TAdj@g - B@u + DN(lambdaa)@divOp@D2N(1./m2)@gradOp@u - TAdj@DN(alpha1)@DN(1./m1)@(T@u-g)
                
                deltaU = sparse.linalg.spsolve(H, F) 
                # deltaU = sparse.linalg.spsolve(H.transpose()@H, H.transpose()@F) 
                deltaP1 = - p1 + DN(1./m1)@DN(alpha1)@(T@(u+deltaU)-g)       - DN(chi1)@DN(1./m1**2)@DN(T@u-g)@T@deltaU@DN(p1)
                deltaP2 = - p2 + D2N(1./m2)@D2N(lambdaa)@gradOp@(u+deltaU) - D2N(chi2)@D2N(1./m2**2)@N(mesh, gradOp@u)@gradOp@deltaU@DN(p2)
                    
                u  = u + deltaU
                p1 = p1 + deltaP1
                p2 = p2 + deltaP2

                if alpha1.all() != 0:
                    p1 = alpha1/np.maximum(alpha1, np.abs(p1))*p1
                p2 = np.hstack( (lambdaa,lambdaa))/np.maximum(np.hstack( (lambdaa,lambdaa)), np.hstack( (rho2N(mesh, p2),rho2N(mesh, p2))))*p2
                
                residual_l2 = self.residual(u, p1, p2)
                if residual_l2 < self.epsilon:
                    break
                bar.text("stopping criterion: "+str(residual_l2))
                bar()
        [err1, err2] = error_indicators.L1L2TV_dualgap_error_scalar(mesh, u, p1, p2, g, lambdaa, alpha1, alpha2, beta, B, T, TAdj, S, gamma1, gamma2, dofs, elements)     
        err = err1 - err2
        return [u, p1, p2, err]

    def __str__(self):
        return "Algorithm: newton\n - max_it  = "+str(self.max_it)+"\n - epsilon = "+str(self.epsilon)

class L1L2TVNewtonOpticalFlow(object):
    def __init__(self, w, h, max_it, epsilon, theta):
        self.w = w
        self.h = h
        self.max_it = max_it
        self.epsilon = epsilon
        self.model = models.L1L2TVModel()
        self.theta = theta

    def init(self, mesh, f0, f1, u0, model):
        dofs = len(mesh.getElements())
        self.mesh = mesh
        self.model = model
        self.gradOp = operators.gradOperator(mesh, bc='N')
        self.divOp = operators.divOperator(mesh, bc='D')
        gradF = operators.gradOperator_centered(mesh, bc='N')@f1
        self.TAdj = bmat([ [ sparse.diags(gradF[0:dofs]) ], [ sparse.diags(gradF[dofs:2*dofs]) ] ])
        self.T = self.TAdj.transpose()
        self.TAdjT = self.TAdj@self.T
        self.S = bmat([ [self.gradOp, sparse.csr_array((2*dofs, dofs))], [sparse.csr_array((2*dofs, dofs)), self.gradOp] ])
        self.SAdj = bmat([ [-self.divOp, sparse.csr_array((dofs, 2*dofs))], [sparse.csr_array((dofs, 2*dofs)), -self.divOp] ])
        self.SAdjS = self.SAdj@self.S # - Laplacian
        self.B = sparse.diags(np.concatenate((self.model.alpha2, self.model.alpha2)))*self.TAdjT + sparse.diags(np.concatenate((self.model.beta, self.model.beta)))@self.SAdjS
        self.g = self.T@u0 - (f1 - f0)

    def residual(self, u, p1, p2):
        mesh = self.mesh
        elements = mesh.getElements()
        dofs = len(elements)
        
        alpha1  = self.model.alpha1
        alpha2  = self.model.alpha2
        lambdaa = self.model.lambdaa
        beta    = self.model.beta
        gamma1  = self.model.gamma1
        gamma2  = self.model.gamma2
        gradOp = self.gradOp
        divOp  = self.divOp
        T = self.T
        B = self.B
        S = self.S
        g = self.g
        TAdj = self.TAdj
        
        rhoN = lambda mesh, v : np.abs( v[0:dofs] )
        rhoF = lambda mesh, A : np.sqrt( A[0:dofs]**2 + A[dofs:2*dofs]**2 + A[2*dofs:3*dofs]**2 + A[3*dofs:4*dofs]**2 )
        DN = lambda v: sparse.diags(v)
        D2N = lambda v: sparse.diags( np.hstack( ( v,v ) ) )
        D4N = lambda v: sparse.diags( np.hstack( ( v,v,v,v ) ) )
        N = lambda mesh, gradUl: bmat([ [ DN(gradUl[0:dofs]), DN(gradUl[dofs:2*dofs]), DN(gradUl[2*dofs:3*dofs]), DN(gradUl[3*dofs:4*dofs]) ], [ DN(gradUl[0:dofs]), DN(gradUl[dofs:2*dofs]), DN(gradUl[2*dofs:3*dofs]), DN(gradUl[3*dofs:4*dofs]) ], [ DN(gradUl[0:dofs]), DN(gradUl[dofs:2*dofs]), DN(gradUl[2*dofs:3*dofs]), DN(gradUl[3*dofs:4*dofs]) ], [ DN(gradUl[0:dofs]), DN(gradUl[dofs:2*dofs]), DN(gradUl[2*dofs:3*dofs]), DN(gradUl[3*dofs:4*dofs]) ] ])
        divOp2N = bmat([ [ divOp, sparse.csr_matrix((dofs, 2*dofs)) ], [ sparse.csr_matrix((dofs, 2*dofs)), divOp ] ])
        gradOp2N = bmat([ [ gradOp, sparse.csr_matrix((2*dofs, dofs)) ], [ sparse.csr_matrix((2*dofs, dofs)), gradOp ] ])

        mK = np.zeros(dofs)
        for e in elements:
            mK[e.indice] = e.dx*e.dy

        m1 = np.maximum( gamma1, rhoN(mesh, T@u-g) )
        m2 = np.maximum( gamma2, rhoF(mesh, gradOp2N@u) )

        if alpha1.all() == 0:
            gamma1 = np.zeros(dofs)
        chi1 = (rhoN(mesh, T@u-g) >= gamma1).astype(int) # gamma1 = 0 if alpha1=0
        chi2 = (rhoF(mesh, gradOp2N@u) >= gamma2).astype(int)
            
        residual_part_1 = TAdj@p1 - divOp2N@p2 - D2N(alpha2)@TAdj@g + B@u
        residual_part_2 = DN(m1)@p1 - DN(alpha1)@(T@u - g)
        residual_part_3 = D4N(m2)@p2 - D4N(lambdaa)@gradOp2N@u
        residual = residual_part_1[0:dofs] + residual_part_1[dofs:2*dofs] + residual_part_2 + residual_part_3[0:dofs] + residual_part_3[dofs:2*dofs] + residual_part_3[2*dofs:3*dofs] + residual_part_3[3*dofs:4*dofs]
        residual_l2 = np.sqrt(np.sum(mK*residual**2))# / self.w / self.h

        return residual_l2

    def run(self): 
        mesh = self.mesh
        elements = mesh.getElements()
        dofs = len(elements)
        
        alpha1  = self.model.alpha1
        alpha2  = self.model.alpha2
        lambdaa = self.model.lambdaa
        beta    = self.model.beta
        gamma1  = self.model.gamma1
        gamma2  = self.model.gamma2
        gradOp = self.gradOp
        divOp  = self.divOp
        T = self.T
        B = self.B
        S = self.S
        g = self.g
        TAdj = self.TAdj
        
        rhoN = lambda mesh, v : np.abs( v[0:dofs] )
        rhoF = lambda mesh, A : np.sqrt( A[0:dofs]**2 + A[dofs:2*dofs]**2 + A[2*dofs:3*dofs]**2 + A[3*dofs:4*dofs]**2 )
        DN = lambda v: sparse.diags(v)
        D2N = lambda v: sparse.diags( np.hstack( ( v,v ) ) )
        D4N = lambda v: sparse.diags( np.hstack( ( v,v,v,v ) ) )
        N = lambda mesh, gradUl: bmat([ [ DN(gradUl[0:dofs]), DN(gradUl[dofs:2*dofs]), DN(gradUl[2*dofs:3*dofs]), DN(gradUl[3*dofs:4*dofs]) ], [ DN(gradUl[0:dofs]), DN(gradUl[dofs:2*dofs]), DN(gradUl[2*dofs:3*dofs]), DN(gradUl[3*dofs:4*dofs]) ], [ DN(gradUl[0:dofs]), DN(gradUl[dofs:2*dofs]), DN(gradUl[2*dofs:3*dofs]), DN(gradUl[3*dofs:4*dofs]) ], [ DN(gradUl[0:dofs]), DN(gradUl[dofs:2*dofs]), DN(gradUl[2*dofs:3*dofs]), DN(gradUl[3*dofs:4*dofs]) ] ])
        divOp2N = bmat([ [ divOp, sparse.csr_matrix((dofs, 2*dofs)) ], [ sparse.csr_matrix((dofs, 2*dofs)), divOp ] ])
        gradOp2N = bmat([ [ gradOp, sparse.csr_matrix((2*dofs, dofs)) ], [ sparse.csr_matrix((2*dofs, dofs)), gradOp ] ])

        mK = np.zeros(dofs)
        for e in elements:
            mK[e.indice] = e.dx*e.dy
        
        # Loop Init
        u  = TAdj@g # np.zeros(2*dofs)
        p1 = T@u # np.zeros(dofs)
        p2 = gradOp2N@u # gradOp2N@u #np.zeros(4*dofs)
        # Loop
        with alive_bar(self.max_it, force_tty=True) as bar:
            for i in range(self.max_it):
                m1 = np.maximum( gamma1, rhoN(mesh, T@u-g) )
                m2 = np.maximum( gamma2, rhoF(mesh, gradOp2N@u) )

                if alpha1.all() == 0:
                    gamma1 = np.zeros(dofs)
                chi1 = (rhoN(mesh, T@u-g) >= gamma1).astype(int) # gamma1 = 0 if alpha1=0
                chi2 = (rhoF(mesh, gradOp2N@u) >= gamma2).astype(int)

                if alpha1.all() != 0:
                    p1 = alpha1/np.maximum(alpha1, np.abs(p1))*p1
                rhoFP2 = rhoF(mesh, p2)
                p2 = np.hstack( (lambdaa,lambdaa,lambdaa,lambdaa))/np.maximum(np.hstack( (lambdaa,lambdaa,lambdaa,lambdaa)), np.hstack( (rhoFP2,rhoFP2,rhoFP2,rhoFP2)))*p2
            
                # gradOp2NU = gradOp2N@u
                # rhoFGradU = rhoF(mesh, gradOp2NU)
                # xHl = []
                # yHl = []
                # vHl = []
                # for e in elements:
                #     if chi2[e.indice] == 1: 
                #         c11 = lambdaa[e.indice] - 1./rhoFGradU[e.indice]*p2[e.indice]*gradOp2NU[e.indice]
                #         c12 = 0                 - 1./rhoFGradU[e.indice]*p2[e.indice]*gradOp2NU[e.indice+dofs]
                #         c13 = 0                 - 1./rhoFGradU[e.indice]*p2[e.indice]*gradOp2NU[e.indice+2*dofs]
                #         c14 = 0                 - 1./rhoFGradU[e.indice]*p2[e.indice]*gradOp2NU[e.indice+3*dofs]
                #         c21 = 0                 - 1./rhoFGradU[e.indice]*p2[e.indice+dofs]*gradOp2NU[e.indice]
                #         c22 = lambdaa[e.indice] - 1./rhoFGradU[e.indice]*p2[e.indice+dofs]*gradOp2NU[e.indice+dofs]
                #         c23 = 0                 - 1./rhoFGradU[e.indice]*p2[e.indice+dofs]*gradOp2NU[e.indice+2*dofs]
                #         c24 = 0                 - 1./rhoFGradU[e.indice]*p2[e.indice+dofs]*gradOp2NU[e.indice+3*dofs]
                #         c31 = 0                 - 1./rhoFGradU[e.indice]*p2[e.indice+2*dofs]*gradOp2NU[e.indice]
                #         c32 = 0                 - 1./rhoFGradU[e.indice]*p2[e.indice+2*dofs]*gradOp2NU[e.indice+dofs]
                #         c33 = lambdaa[e.indice] - 1./rhoFGradU[e.indice]*p2[e.indice+2*dofs]*gradOp2NU[e.indice+2*dofs]
                #         c34 = 0                 - 1./rhoFGradU[e.indice]*p2[e.indice+2*dofs]*gradOp2NU[e.indice+3*dofs]
                #         c41 = 0                 - 1./rhoFGradU[e.indice]*p2[e.indice+3*dofs]*gradOp2NU[e.indice]
                #         c42 = 0                 - 1./rhoFGradU[e.indice]*p2[e.indice+3*dofs]*gradOp2NU[e.indice+dofs]
                #         c43 = 0                 - 1./rhoFGradU[e.indice]*p2[e.indice+3*dofs]*gradOp2NU[e.indice+2*dofs]
                #         c44 = lambdaa[e.indice] - 1./rhoFGradU[e.indice]*p2[e.indice+3*dofs]*gradOp2NU[e.indice+3*dofs]
                #     else:
                #         c11 = lambdaa[e.indice]
                #         c12 = 0
                #         c13 = 0
                #         c14 = 0
                #         c21 = 0
                #         c22 = lambdaa[e.indice]
                #         c23 = 0
                #         c24 = 0
                #         c31 = 0
                #         c32 = 0
                #         c33 = lambdaa[e.indice]
                #         c34 = 0
                #         c41 = 0
                #         c42 = 0
                #         c43 = 0
                #         c44 = lambdaa[e.indice]
                #     c = np.array([[c11, c12, c13, c14], [c21, c22, c23, c24], [c31, c32, c33, c34], [c41, c42, c43, c44]])
                #     eigenvalues, eigenvectors = np.linalg.eigh(0.5*(c+c.transpose()))
                #     cHat = eigenvectors @ np.diag(np.maximum(eigenvalues, 0)) @ np.linalg.inv(eigenvectors) # projection to positive definite matrices space.
                #     xHl.append(e.indice)
                #     yHl.append(e.indice)
                #     vHl.append(cHat[0,0])
                #     xHl.append(e.indice + dofs)
                #     yHl.append(e.indice)
                #     vHl.append(cHat[0,1])
                #     xHl.append(e.indice + 2*dofs)
                #     yHl.append(e.indice)
                #     vHl.append(cHat[0,2])
                #     xHl.append(e.indice + 3*dofs)
                #     yHl.append(e.indice)
                #     vHl.append(cHat[0,3])
                #     xHl.append(e.indice)
                #     yHl.append(e.indice + dofs)
                #     vHl.append(cHat[1,0])
                #     xHl.append(e.indice + dofs)
                #     yHl.append(e.indice + dofs)
                #     vHl.append(cHat[1,1])
                #     xHl.append(e.indice + 2*dofs)
                #     yHl.append(e.indice + dofs)
                #     vHl.append(cHat[1,2])
                #     xHl.append(e.indice + 3*dofs)
                #     yHl.append(e.indice + dofs)
                #     vHl.append(cHat[1,3])
                #     xHl.append(e.indice)
                #     yHl.append(e.indice + 2*dofs)
                #     vHl.append(cHat[2,0])
                #     xHl.append(e.indice + dofs)
                #     yHl.append(e.indice + 2*dofs)
                #     vHl.append(cHat[2,1])
                #     xHl.append(e.indice + 2*dofs)
                #     yHl.append(e.indice + 2*dofs)
                #     vHl.append(cHat[2,2])
                #     xHl.append(e.indice + 3*dofs)
                #     yHl.append(e.indice + 2*dofs)
                #     vHl.append(cHat[2,3])
                #     xHl.append(e.indice)
                #     yHl.append(e.indice + 3*dofs)
                #     vHl.append(cHat[3,0])
                #     xHl.append(e.indice + dofs)
                #     yHl.append(e.indice + 3*dofs)
                #     vHl.append(cHat[3,1])
                #     xHl.append(e.indice + 2*dofs)
                #     yHl.append(e.indice + 3*dofs)
                #     vHl.append(cHat[3,2])
                #     xHl.append(e.indice + 3*dofs)
                #     yHl.append(e.indice + 3*dofs)
                #     vHl.append(cHat[3,3])
                # Cl = sparse.csr_matrix((vHl, (yHl, xHl)), shape=[4*dofs, 4*dofs])
                Cl = D4N(lambdaa) - D4N(chi2)@DN(p2)@D4N(1./m2)@N(mesh, gradOp2N@u)
                H = B - divOp2N@D4N(1./m2)@Cl@gradOp2N + TAdj@DN(1./m1)@( DN(alpha1) - DN(chi1)@DN(1./m1)@DN(T@u-g)@DN(p1) )@T
                F = D2N(alpha2)@TAdj@g - B@u + D2N(lambdaa)@divOp2N@D4N(1./m2)@gradOp2N@u - TAdj@DN(alpha1)@DN(1./m1)@(T@u-g)
                
                deltaU  = sparse.linalg.spsolve(H, F) 
                # deltaU  = sparse.linalg.spsolve(H.transpose()@H, H.transpose()@F) 
                deltaP1 = - p1 + DN(1./m1)@DN(alpha1)@(T@(u+deltaU)-g)       - DN(chi1)@DN(1./m1**2)@DN(T@u-g)@DN(p1)@T@deltaU
                deltaP2 = - p2 + D4N(1./m2)@D4N(lambdaa)@gradOp2N@(u+deltaU) - D4N(chi2)@D4N(1./m2**2)@DN(p2)@N(mesh, gradOp2N@u)@gradOp2N@deltaU

                u = u + deltaU
                p1 = p1 + deltaP1
                p2 = p2 + deltaP2

                if alpha1.all() != 0:
                    p1 = alpha1/np.maximum(alpha1, np.abs(p1))*p1
                rhoFP2 = rhoF(mesh, p2)
                p2 = np.hstack( (lambdaa,lambdaa,lambdaa,lambdaa))/np.maximum(np.hstack( (lambdaa,lambdaa,lambdaa,lambdaa)), np.hstack( (rhoFP2,rhoFP2,rhoFP2,rhoFP2)))*p2

                residual_l2 = self.residual(u, p1, p2)
                if residual_l2 < self.epsilon:
                    break
                    
                bar.text("stopping criterion: "+str(residual_l2))
                bar()
        [err1, err2] = error_indicators.L1L2TV_dualgap_error(mesh, u, p1, p2, g, lambdaa, alpha1, alpha2, beta, B, T, TAdj, S, gamma1, gamma2, dofs, elements)     
        err = err1 - err2
        return [u, p1, p2, err]

    def __str__(self):
        return "Algorithm: newton\n - max_it  = "+str(self.max_it)+"\n - epsilon = "+str(self.epsilon)+"\n - theta = "+str(self.theta)