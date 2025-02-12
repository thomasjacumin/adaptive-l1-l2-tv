import numpy as np
import projection
from PIL import Image

import matplotlib.pyplot as plt

import quadmesh
import models
import newton

r = 1.5
max_n = 11
w = 2**max_n
h = w

# # Model
# model = models.L1L2TVModel()
# model.alpha1  = 1*np.ones(w*h)
# model.alpha2  = 1*np.ones(w*h)
# model.lambdaa = 1*np.ones(w*h)
# model.beta    = 0*np.ones(w*h)
# model.gamma1  = 2e-4*np.ones(w*h)
# model.gamma2  = 2e-4*np.ones(w*h)

# Model
model = models.L1L2TVModel()
model.alpha1  = 1*np.ones(w*h)
model.alpha2  = 1*np.ones(w*h)
model.lambdaa = 1*np.ones(w*h)
model.beta    = 0*np.ones(w*h)
model.gamma1  = 1e-5*np.ones(w*h)
model.gamma2  = 1e-5*np.ones(w*h)

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

# exact solution
u_exact = np.zeros(dofs)
if r < 1:
    u_exact = np.zeros(dofs)
else:
    u_exact = 2*( 1 - 1/(model.alpha2*r))*f
Image.fromarray( np.uint8(255*f).reshape([h,w]), 'L' ).save("results/convergence/f.png")
Image.fromarray( np.uint8(255*u_exact).reshape([h,w]), 'L' ).save("results/convergence/u_exact.png")

########################################

list_dofs_unif = []
list_error_unif = []

elements = viewExact.getElements()
dofs = len(elements)

algorithm = newton.L1L2TVNewtonDenoising(10000, 1e-6)

print(max_n)
for n in range(2, max_n-1):
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

    mK = np.zeros(dofs)
    for e in elements:
        mK[e.indice] = e.dx*e.dy
    
    PInv = projection.projectionMatrix(view, viewExact)
    l2_error = np.sqrt( np.sum( mK*(u_exact - PInv@u)**2 ) ) 
    print(l2_error)

    elements_curr = view.getElements()
    dofs_curr = len(elements_curr)
    list_dofs_unif.append(dofs_curr)
    list_error_unif.append(l2_error)
    print(n, dofs_curr)

######################################################

def adaptMesh(view, err, model, u):
    elements = view.getElements()
    dofs = len(elements)
    
    iSorted = np.argsort(err)[::-1]

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

############################################################

theta_mark = 0.4
N = (max_n - 2) - 2

list_dofs_ada_1 = []
list_error_ada_1 = []

algorithm = newton.L1L2TVNewtonDenoising(10000, 1e-6)

view = quadmesh.QuadMeshLeafView(6, 6, int(w/2**(N)), int(h/2**(N)))
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
mK = np.zeros(dofs)
for e in elements:
    mK[e.indice] = e.dx*e.dy

for n in range(0, N-1):
    elements_curr = view.getElements()
    dofs_curr = len(elements_curr)

    print(n, dofs_curr)
    #quadmesh.showQMeshFunction(view, np.ones(dofs_curr), pathname="results/convergence/mesh-"+str(n)+".png")
    
    algorithm.init(view, g, model_approx)
    [u, p1, p2, err] = algorithm.run()
    print(np.min(err), np.max(err), np.sum(err))
    err = err - np.min(err)

    print("compute l2 error...")    
    PInv = projection.projectionMatrix(view, viewExact)
    l2_error = np.sqrt( np.sum( mK*(u_exact - PInv@u)**2 ) ) 
    print(l2_error)
    list_dofs_ada_1.append(dofs_curr)
    list_error_ada_1.append(l2_error)
  
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

#########################################

N = (max_n - 2)

list_dofs_ada_2 = []
list_error_ada_2 = []

algorithm = newton.L1L2TVNewtonDenoising(10000, 1e-6)

view = quadmesh.QuadMeshLeafView(6, 6, int(w/2**(N)), int(h/2**(N)))
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
mK = np.zeros(dofs)
for e in elements:
    mK[e.indice] = e.dx*e.dy

for n in range(0, N):
    elements_curr = view.getElements()
    dofs_curr = len(elements_curr)

    print(n, dofs_curr)
    
    algorithm.init(view, g, model_approx)
    [u, p1, p2, err] = algorithm.run()
    print(np.min(err), np.max(err), np.sum(err))
    err = err - np.min(err)

    print("compute l2 error...")    
    PInv = projection.projectionMatrix(view, viewExact)
    l2_error = np.sqrt( np.sum( mK*(u_exact - PInv@u)**2 ) ) 
    print(l2_error)
    list_dofs_ada_2.append(dofs_curr)
    list_error_ada_2.append(l2_error)
    
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

#########################################################################################

guide1_x = np.linspace(list_dofs_unif[1], list_dofs_unif[len(list_dofs_unif)-3], 20)
guide1_y = 3*np.power(guide1_x, -1/4)
guide2_x = np.linspace(list_dofs_ada_1[0], list_dofs_ada_1[len(list_dofs_ada_1)-2], 20)
guide2_y = 8*np.power(guide2_x, -1/2)

plt.plot(guide1_x, guide1_y, 'k:', label=r"$\sharp dofs^{-1/4}$")
plt.plot(guide2_x, guide2_y, 'k--', label=r"$\sharp dofs^{-1/2}$")

plt.plot(list_dofs_unif[1:len(list_dofs_unif)], list_error_unif[1:len(list_dofs_unif)], 'x-', label="uniform")
plt.plot(list_dofs_ada_2, list_error_ada_2, 's-', label="adaptative")
plt.plot(list_dofs_ada_1, list_error_ada_1, 'o-', label="adaptative")

plt.ylim(0.1, 1)
plt.xscale("log")
plt.yscale("log")
plt.grid(True, which='both', linestyle='--', color='gray', alpha=0.4)

plt.xlabel(r"$\sharp dofs$")
plt.ylabel(r"$\|u_\text{exact} - I_h u_h\|_{L^2(\Omega)}$")
plt.title("convergence")
plt.legend()
plt.savefig('results/convergence/convergence.png')
plt.show()