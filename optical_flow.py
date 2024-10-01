import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

import utils
import newton
import data
import models
import runners
import warp

# Parse parameters
parser = argparse.ArgumentParser(description="sample argument parser")
parser.add_argument("f0", help="first frame")
parser.add_argument("f1", help="second frame")
parser.add_argument("--out", nargs='?', help="optical flow output")
parser.add_argument("--ground-truth", nargs='?', help="optical flow ground truth")
parser.add_argument("--save-benchmark", nargs='?', help="file output of benchmark")
# Model parameters
parser.add_argument("--alpha1", nargs='?', type=float, default=0., help="parameter for the optical flow model")
parser.add_argument("--alpha2", nargs='?', type=float, default=0., help="parameter for the optical flow model")
parser.add_argument("--lambdaa", nargs='?', type=float, default=1., help="parameter for the optical flow model")
parser.add_argument("--beta", nargs='?', type=float, default=0., help="parameter for the optical flow model")
parser.add_argument("--gamma1", nargs='?', type=float, default=0., help="parameter for the optical flow model")
parser.add_argument("--gamma2", nargs='?', type=float, default=0., help="parameter for the optical flow model")
# Algorithm parameters
parser.add_argument("--algo", type=str, choices=["chambolle", "chambolle_pock", "zach_pock", "newton", "horn_schunk"], help="algorithm to compute the optical flow")
parser.add_argument("--algo-max-it", nargs='?', type=int, default=0, help="maximal number of iteration for the optical flow algorithm")
parser.add_argument("--algo-epsilon", nargs='?', type=float, default=-1., help="stopping threshold for the optical flow algorithm")
parser.add_argument("--algo-theta", nargs='?', type=float, default=1., help="theta in the semismooth newton agorithm")
parser.add_argument("--T-scheme", type=str, choices=["forward", "backward", "centered"], help="finite differences scheme for the operator T")
# Coarse-to-fine parameters
parser.add_argument("--N-refine", nargs='?', type=int, default=0, help="number of refinement allowed (0 for non-adaptivity)")
parser.add_argument("--ctf-epsilon", nargs='?', type=float, default=-1., help="stopping threshold for the coarse-to-fine algorithm")
parser.add_argument("--ctf-max-dofs", nargs='?', type=float, default=1., help="maximal number of elements (%)")
parser.add_argument("--ctf-max-new-dofs", nargs='?', type=float, default=0.1, help="maximal number of new elements per refinment (%)")
parser.add_argument("--ctf-uniform", action=argparse.BooleanOptionalAction, help="Only use uniform mesh")
parser.add_argument("--ctf-lambda-adaptivity", action=argparse.BooleanOptionalAction, help="Enable lambda automatic selection")
parser.add_argument("--ctf-warp", action=argparse.BooleanOptionalAction, help="Enable warping with doing coarse-to-fine")
parser.add_argument("--ctf-mark", nargs='?', type=float, default=0.75, help="Proportion of elements to refine at each iteration")
# Warping parameters
parser.add_argument("--warp", action=argparse.BooleanOptionalAction, help="warping")
parser.add_argument("--warp-epsilon", nargs='?', type=float, default=-1., help="stopping threshold for the warping algorithm")
parser.add_argument("--warp-max-it", nargs='?', type=int, default=10, help="maximal number of iteration for the warping algorithm")
# Other parameters
parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, help="normalize the input images and the warping if enabled")

args = parser.parse_args()

# Create data
f0, w, h = data.image(args.f0)
f1, w, h = data.image(args.f1)

print("***********************************")
print("Input images: ")
print(" - f0 = "+str(args.f0))
print(" - f1 = "+str(args.f1))
if args.normalize == True:
    print(" - normalize input images")
    f0 = f0/(np.sum(f0)/(w*h))
    f1 = f1/(np.sum(f1)/(w*h))

# Create model
model = models.L1L2TVModel()
model.alpha1  = args.alpha1*np.ones(w*h)
model.alpha2  = args.alpha2*np.ones(w*h)
model.lambdaa = args.lambdaa*np.ones(w*h)
model.beta    = args.beta*np.ones(w*h)
model.gamma1  = args.gamma1*np.ones(w*h)
model.gamma2  = args.gamma2*np.ones(w*h)
print(model)

# Start timer
start_time = time.time()

# Create algorithm to solve ROF model
if args.algo == "newton":
    algo_max_it = 30 if args.algo_max_it == 0 else args.algo_max_it
    algo_epsilon = 1e-3 if args.algo_epsilon == -1 else args.algo_epsilon
    algorithm = newton.L1L2TVNewtonOpticalFlow(w, h, algo_max_it, algo_epsilon, args.algo_theta)
else:
    raise NotImplementedError
print(algorithm)

# Create runner for mesh adaptivity (or not)
if args.N_refine > 0:
    ctf_epsilon = 0.5 if args.ctf_epsilon == -1 else args.ctf_epsilon
    if args.ctf_uniform:
        runner = runners.UniformCoarseToFineRunner(algorithm, w, h, args.N_refine, ctf_epsilon, args.ctf_lambda_adaptivity)
    else:
        if args.ctf_warp:
            runner = runners.AllInOneRunner(algorithm, w, h, args.N_refine, ctf_epsilon, args.ctf_mark)
        else:
            runner = runners.CoarseToFineRunner(algorithm, w, h, args.N_refine, ctf_epsilon, args.ctf_max_dofs, args.ctf_max_new_dofs, args.ctf_lambda_adaptivity)
else:
    runner = runners.SimpleRunner(algorithm, w, h)
print(runner)
        
# Warping (or not)
if args.warp and not args.ctf_warp:
    warp_epsilon = 1e-3 if args.warp_epsilon == -1 else args.warp_epsilon
    warping = warp.Warping(runner, f0, f1, w, h, warp_epsilon, args.warp_max_it, args.normalize)
    warping.init(model)
    print(warping)
    [u, p1, p2] = warping.run()
else:
    runner.init(model, f0, f1)
    [u, p1, p2] = runner.run()

# stop timer
timer = time.time() - start_time

if args.ground_truth:
    print("Benchmark:")
    wGT, hGT, uGT, vGT = utils.openFlo(args.ground_truth)
    assert(wGT == w and hGT == h)
    AEE, SDEE = utils.EE(w, h, u[0:w*h], u[w*h:2*w*h], uGT, vGT)
    AAE, SDAE = utils.AE(w, h, u[0:w*h], u[w*h:2*w*h], uGT, vGT)
    print(" - EE-mean: "+str(AEE))
    print(" - EE-stddev: "+str(SDEE))
    print(" - AE-mean: "+str(AAE))
    print(" - AE-stddev: "+str(SDAE))
    print(" - time: "+str(timer)+"s")

    if args.save_benchmark:
        f = open(args.save_benchmark, "w")
        f.write("EE-mean: "+str(AEE)+"\n")
        f.write("EE-stddev: "+str(SDEE)+"\n")
        f.write("AE-mean: "+str(AAE)+"\n")
        f.write("AE-stddev: "+str(SDAE)+"\n")
        f.write("time: "+str(timer)+"s")
        f.close()

if args.out:
    print("saving flo file...")
    utils.saveFlo(w, h, u[0:w*h], u[w*h:2*w*h], args.out)
    
print("***********************************")