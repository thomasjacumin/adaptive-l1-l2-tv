import numpy as np
import projection
from PIL import Image

import matplotlib.pyplot as plt

import quadmesh
import models
import newton

r = 1.5
w = 512
h = w
theta_mark = 0.2

# Model
model = models.L1L2TVModel()
model.alpha1  = 1*np.ones(w*h)
model.alpha2  = 1*np.ones(w*h)
model.lambdaa = 1*np.ones(w*h)
model.beta    = 0*np.ones(w*h)
model.gamma1  = 2e-4*np.ones(w*h)
model.gamma2  = 2e-4*np.ones(w*h)

# Mesh
viewExact = quadmesh.QuadMeshLeafView(6, 6, w, h)
viewExact.create()
viewExact.computeIndices()
elements = viewExact.getElements()
dofs = len(elements)

# data
f = np.zeros(dofs)
for e in elements:
    if (e.x - 3)**2 + (e.y - 3)**2 <= r**2:
        f[e.indice] = 1

max_n = int(np.log(w))

# exact solution
u_exact = np.zeros(dofs)
if r < 1:
    u_exact = np.zeros(dofs)
else:
    u_exact = 2*( 1 - 1/(model.alpha2*r))*f
Image.fromarray( np.uint8(255*f).reshape([h,w]), 'L' ).save("results/f.png")
Image.fromarray( np.uint8(255*u_exact).reshape([h,w]), 'L' ).save("results/u_exact.png")





list_dofs_unif = []
list_error_unif = []

elements = viewExact.getElements()
dofs = len(elements)

algorithm = newton.L1L2TVNewtonDenoising(100, 1e-3)

print(max_n)
for n in range(2, max_n):
    view = quadmesh.QuadMeshLeafView(6, 6, int(w/2**n), int(h/2**n))
    view.create()
    view.computeIndices()
    P = projection.projectionMatrix(viewExact, view)
    
    model_approx = models.L1L2TVModel()
    model_approx.alpha1  = P@model.alpha1
    model_approx.alpha2  = P@model.alpha2
    model_approx.lambdaa = P@model.lambdaa
    model_approx.beta    = P@model.beta
    model_approx.gamma1  = P@model.gamma1
    model_approx.gamma2  = P@model.gamma2
    
    algorithm.init(view, P@f, model_approx)
    [u, p1, p2, err] = algorithm.run()
    
    # quadmesh.showQMeshFunction(view, u, displayEdges=False, vmin=0, vmax=1)
    # print(np.min(u), np.max(u))

    mK = np.zeros(dofs)
    for e in elements:
        mK[e.indice] = e.dx*e.dy
    
    PInv = projection.projectionMatrix(view, viewExact)
    l2_error = np.sqrt( np.sum( mK*(u_exact - PInv@u)**2 ) ) 
    print(l2_error)
    # print(np.min(PInv@u), np.max(PInv@u))

    # elements = view.getElements()
    # dofs = len(elements)

    Image.fromarray( np.uint8(255*PInv@u).reshape([h,w]), 'L' ).save("results/unif-"+str(n)+".png")

    elements_curr = view.getElements()
    dofs_curr = len(elements_curr)
    list_dofs_unif.append(dofs_curr)
    list_error_unif.append(l2_error)





def adaptMesh(view, err, model, u):
    elements = view.getElements()
    dofs = len(elements)
    
    iSorted = np.argsort(err)[::-1]

    #print(err[iSorted])

    # Mark elements
    elementsToRefine = []
    sum_eta = np.sum(err)
    current_sum_eta = 0
    for n in range(0, dofs):
        e = elements[iSorted[n]]
        if e.level < max_n:
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
    view.balance()
    view.computeIndices()

list_dofs_ada = []
list_error_ada = []

algorithm = newton.L1L2TVNewtonDenoising(100, 1e-3)

view = quadmesh.QuadMeshLeafView(6, 6, int(w/2**(max_n-1)), int(h/2**(max_n-1)))
view.create()
view.computeIndices()

P = projection.projectionMatrix(viewExact, view)

model_approx = models.L1L2TVModel()
model_approx.alpha1  = P@model.alpha1
model_approx.alpha2  = P@model.alpha2
model_approx.lambdaa = P@model.lambdaa
model_approx.beta    = P@model.beta
model_approx.gamma1  = P@model.gamma1
model_approx.gamma2  = P@model.gamma2

g = P@f

elements = viewExact.getElements()
dofs = len(elements)

for n in range(0, max_n+1):
    elements_curr = view.getElements()
    dofs_curr = len(elements_curr)
    
    quadmesh.showQMeshFunction(view, np.ones(dofs_curr), pathname="results/mesh-"+str(n)+".png")
    
    algorithm.init(view, g, model_approx)
    [u, p1, p2, err] = algorithm.run()
    print(np.min(err), np.max(err), np.sum(err))
    err = err - np.min(err)

    mK = np.zeros(dofs)
    for e in elements:
        mK[e.indice] = e.dx*e.dy
        
    PInv = projection.projectionMatrix(view, viewExact)
    l2_error = np.sqrt( np.sum( mK*(u_exact - PInv@u)**2 ) ) 
    print(l2_error)

    elements_curr = view.getElements()
    dofs_curr = len(elements_curr)
    list_dofs_ada.append(dofs_curr)
    list_error_ada.append(l2_error)

    # Image.fromarray( np.uint8(255*PInv@u).reshape([h,w]), 'L' ).save("results/ada-"+str(n)+".png")

    print("adapt mesh...")
    adaptMesh(view, err, model_approx, u)
    
    print("compute projections...")
    P = projection.projectionMatrix(viewExact, view)
    model_approx.alpha1  = P@model.alpha1
    model_approx.alpha2  = P@model.alpha2
    model_approx.lambdaa = P@model.lambdaa
    model_approx.beta    = P@model.beta
    model_approx.gamma1  = P@model.gamma1
    model_approx.gamma2  = P@model.gamma2
    g = P@f


f = open("results/convergence.txt", "w")
f.write(str(list_dofs_unif)+"\n")
f.write(str(list_error_unif)+"\n")

f.write(str(list_dofs_ada)+"\n")
f.write(str(list_error_ada))
f.close()





plt.plot(np.log10(list_dofs_unif), list_error_unif, label="uniform")
plt.plot(np.log10(list_dofs_ada), list_error_ada, label="adaptative")

plt.xlabel("log(#dofs) ")
plt.ylabel("$\\|u_\\text{exact} - u_h\\|_{L^2(\Omega)}$")
plt.title("title")
plt.legend()
plt.savefig('results/convergence.png')
plt.show()