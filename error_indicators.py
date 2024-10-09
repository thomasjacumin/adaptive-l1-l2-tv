import numpy as np
from scipy import sparse
from scipy.sparse import bmat
import operators
import quadmesh

# def L1L2TV_dualgap_error_scalar(mesh, v, q1, q2, g, lambdaa, alpha1, alpha2, beta, B, T, TAdj, S, gamma1, gamma2, dofs, elements):
#     elements = mesh.getElements()
#     dofs = len(elements)

#     gradOp = operators.gradOperator(mesh, bc="N")
#     divOp  = operators.divOperator(mesh, bc="D")
    
#     hubert = lambda x, gamma: 1./(2*gamma)*x**2 if np.abs(x) <= gamma else np.abs(x) - gamma/2
#     hubert_vec = np.vectorize(hubert)
#     rho2N = lambda mesh, v : np.sqrt( v[0:dofs]**2 + v[dofs:2*dofs]**2 )
    
#     Tv = T@v
#     Sv = S@v
#     gradV = gradOp@v
#     TAdj_q1 = TAdj@q1
#     TAdj_g = TAdj@g
#     L2DualTerms = TAdj_q1 - divOp@q2 - alpha2*TAdj_g
#     vFromDualVariable = sparse.linalg.spsolve(B, L2DualTerms)  
   
#     term2  = np.zeros(dofs)
#     term3  = np.zeros(dofs)
#     term4  = np.zeros(dofs)
#     term1 = np.zeros(dofs)
#     term5 = np.zeros(dofs)
#     term7 = np.zeros(dofs)
#     term4 = np.zeros(dofs)

#     mK = np.zeros(dofs)
#     for e in elements:
#         mK[e.indice] = e.dx*e.dy

#     ## F1*
#     if alpha1.all() > 0:
#         term1 = mK*alpha1*hubert_vec(Tv-g, gamma1)
#     term2 = -mK*(Tv-g)*q1
#     ## L2-Regul p1
#     if alpha1.all() > 0:
#         term3 = mK*gamma1/(2*alpha1)*q1**2
#     ## F2*
#     if lambdaa.all() > 0:
#         term4 = mK*lambdaa*hubert_vec(rho2N(mesh, gradV), gamma2)
#     term5 = -mK*( gradV[0:dofs]*q2[0:dofs] + gradV[dofs:2*dofs]*q2[dofs:2*dofs] )
#     ## L2-Regul p2
#     if lambdaa.all() > 0:
#         term6 = mK*gamma2/(2*lambdaa)*(q2[0:dofs]**2 + q2[dofs:2*dofs]**2)
#     # term7 = 0.5*mK*(vFromDualVariable + v)*(B@(vFromDualVariable + v))
#     term7 = 0.5*mK*(-vFromDualVariable - v)**2

#     # quadmesh.showQMeshFunction(mesh, term1, displayEdges=False)
#     # quadmesh.showQMeshFunction(mesh, term2, displayEdges=False)
#     # quadmesh.showQMeshFunction(mesh, term3, displayEdges=False)
#     # quadmesh.showQMeshFunction(mesh, term4, displayEdges=False)
#     # quadmesh.showQMeshFunction(mesh, term5, displayEdges=False)
#     # quadmesh.showQMeshFunction(mesh, term6, displayEdges=False)
#     # quadmesh.showQMeshFunction(mesh, term7, displayEdges=False)
                                                                                    
#     return [term1 + term2 + term3 + term4 + term5 + term6 + term7, np.zeros(dofs)]

def L1L2TV_dualgap_error_scalar(mesh, v, q1, q2, g, lambdaa, alpha1, alpha2, beta, B, T, TAdj, S, gamma1, gamma2, dofs, elements):
    elements = mesh.getElements()
    dofs = len(elements)

    gradOp = operators.gradOperator(mesh, bc="N")
    divOp  = operators.divOperator(mesh, bc="D")
    
    hubert = lambda x, gamma: 1./(2*gamma)*x**2 if np.abs(x) <= gamma else np.abs(x) - gamma/2
    hubert_vec = np.vectorize(hubert)
    rho2N = lambda mesh, v : np.sqrt( v[0:dofs]**2 + v[dofs:2*dofs]**2 )
    
    Tv = T@v
    Sv = S@v
    gradV = gradOp@v
    TAdj_q1 = TAdj@q1
    TAdj_g = TAdj@g
    L2DualTerms = TAdj_q1 - divOp@q2 - alpha2*TAdj_g
    vFromDualVariable = sparse.linalg.spsolve(B, L2DualTerms)  
   
    term2  = np.zeros(dofs)
    term3  = np.zeros(dofs)
    term4  = np.zeros(dofs)
    term8  = np.zeros(dofs)
    term1 = np.zeros(dofs)
    term5 = np.zeros(dofs)
    term7 = np.zeros(dofs)
    term4 = np.zeros(dofs)
    term9  = np.zeros(dofs)

    mK = np.zeros(dofs)
    for e in elements:
        mK[e.indice] = e.dx*e.dy

    ## F1*
    if alpha1.all() > 0:
        term1 = mK*alpha1*hubert_vec(Tv-g, gamma1)
    ## L2-data ter
    term2 = mK*alpha2/2*(Tv-g)**2
    ## L2-Regul S
    term3 = mK*beta/2*Sv**2
    ## F2*
    if lambdaa.all() > 0:
        term4 = mK*lambdaa*hubert_vec(rho2N(mesh, gradV), gamma2)
    
    # Dual
    ## B^-1-Data term
    term5 = -mK/2*L2DualTerms*vFromDualVariable
    ## g**2
    term6 = mK*alpha2/2*g**2
    ## g.p1
    term7 = -mK*g*q1
    ## L2-Regul p1
    if alpha1.all() > 0:
        term8 = -mK*gamma1/(2*alpha1)*q1**2
    ## L2-Regul p2
    if lambdaa.all() > 0:
        term9 = -mK*gamma2/(2*lambdaa)*(q2[0:dofs]**2 + q2[dofs:2*dofs]**2)
    # quadmesh.showQMeshFunction(mesh, -term5, displayEdges=False)
    # quadmesh.showQMeshFunction(mesh, -term7, displayEdges=False)
    # quadmesh.showQMeshFunction(mesh, -term6, displayEdges=False)
                                                                                           
    return [term1 + term2 + term3 + term4, term5 + term6 + term7 + term8 + term9]

def L1L2TV_dualgap_error(mesh, v, q1, q2, g, lambdaa, alpha1, alpha2, beta, B, T, TAdj, S, gamma1, gamma2, dofs, elements):
    elements = mesh.getElements()
    dofs = len(elements)

    gradOp = operators.gradOperator(mesh, bc="N")
    divOp  = operators.divOperator(mesh, bc="D")

    gradOp2N = bmat([ [ gradOp, sparse.csr_matrix((2*dofs, dofs)) ], [ sparse.csr_matrix((2*dofs, dofs)), gradOp ] ])
    divOp2N = bmat([ [ divOp, sparse.csr_matrix((dofs, 2*dofs)) ], [ sparse.csr_matrix((dofs, 2*dofs)), divOp ] ])
    
    hubert = lambda x, gamma: 1./(2*gamma)*x**2 if np.abs(x) <= gamma else np.abs(x) - gamma/2
    hubert_vec = np.vectorize(hubert)
    rhoF = lambda mesh, A : np.sqrt( A[0:dofs]**2 + A[dofs:2*dofs]**2 + A[2*dofs:3*dofs]**2 + A[3*dofs:4*dofs]**2 )
    
    Tv = T@v
    Sv = S@v
    gradV = gradOp2N@v
    TAdj_q1 = TAdj@q1
    TAdj_g = TAdj@g
    L2DualTerms = TAdj_q1 - divOp2N@q2 - np.hstack((alpha2,alpha2))*TAdj_g
    vFromDualVariable = sparse.linalg.spsolve(B, L2DualTerms)  
   
    term2  = np.zeros(dofs)
    term3  = np.zeros(dofs)
    term4  = np.zeros(dofs)
    term8  = np.zeros(dofs)
    term10 = np.zeros(dofs)
    term5 = np.zeros(dofs)
    term6 = np.zeros(dofs)
    term7 = np.zeros(dofs)
    term11 = np.zeros(dofs)
    term9  = np.zeros(dofs)

    mK = np.zeros(dofs)
    for e in elements:
        mK[e.indice] = e.dx*e.dy

    ## L2-data ter
    term2 = mK*alpha2/2*(Tv-g)**2
    ## L2-Regul S
    term5 = mK*beta/2*rhoF(mesh, Sv)**2
    ## F2*
    if lambdaa.all() > 0:
        term11 = mK*lambdaa*hubert_vec(rhoF(mesh, gradV), gamma2)
    ## F1*
    if alpha1.all() > 0:
        term10 = mK*alpha1*hubert_vec(Tv-g, gamma1)

    # Dual
    ## B^-1-Data term
    term7 = -mK/2*(L2DualTerms[0:dofs]*vFromDualVariable[0:dofs] + L2DualTerms[dofs:2*dofs]*vFromDualVariable[dofs:2*dofs])
    ## g**2
    term3 = mK*alpha2/2*g**2
    ## L2-Regul p2
    if lambdaa.all() > 0:
        term9 = -mK*gamma2/(2*lambdaa)*(q2[0:dofs]**2 + q2[dofs:2*dofs]**2) - mK*gamma2/(2*lambdaa)*(q2[2*dofs:3*dofs]**2 + q2[3*dofs:4*dofs]**2)        
    ## g.p1
    term4 = -mK*g*q1
    ## L2-Regul p1
    if alpha1.all() > 0:
        term8 = -mK*gamma1/(2*alpha1)*q1**2
                                                                                           
    return [term2 + term5 + term11 + term10, term7 + term3 + term4 + term8 + term9]
