import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import quadmesh
import newton
import data
import models
import runners

# Parse parameters
parser = argparse.ArgumentParser(description="sample argument parser")
parser.add_argument("g", help="input image")
parser.add_argument("--crop", action=argparse.BooleanOptionalAction, help="crop the input images to have length of a power of 2")
parser.add_argument("--g-out", nargs='?', help="g")
parser.add_argument("--noisy-out", nargs='?', help="noisy image")
parser.add_argument("--out", nargs='?', help="image")
parser.add_argument("--sigma", nargs='?', type=float, default=0., help="noise level (std deviation)")
parser.add_argument("--save-benchmark", nargs='?', help="file output of benchmark")
# Model parameters
parser.add_argument("--alpha1", nargs='?', type=float, default=0., help="parameter for the denoising model")
parser.add_argument("--alpha2", nargs='?', type=float, default=0., help="parameter for the denoising model")
parser.add_argument("--lambdaa", nargs='?', type=float, default=1., help="parameter for the denoising model")
parser.add_argument("--beta", nargs='?', type=float, default=0., help="parameter for the denoising model")
parser.add_argument("--gamma1", nargs='?', type=float, default=0., help="parameter for the denoising model")
parser.add_argument("--gamma2", nargs='?', type=float, default=0., help="parameter for the denoising model")
# Algorithm parameters
parser.add_argument("--algo-max-it", nargs='?', type=int, default=0, help="maximal number of iteration for the denoising algorithm")
parser.add_argument("--algo-epsilon", nargs='?', type=float, default=-1., help="stopping threshold for the denoising algorithm")
# Coarse-to-fine parameters
parser.add_argument("--N-refine", nargs='?', type=int, default=0, help="number of refinement allowed (0 for non-adaptivity)")

args = parser.parse_args()
np.random.seed(0)

# Create data
g_original, w, h = data.image(args.g)
N = min( int(np.log2(w)), int(np.log2(h)) )
if args.crop:
    box = (int(w/2)-2**(N-1), int(h/2)-2**(N-1), int(w/2)+2**(N-1), int(h/2)+2**(N-1))
    g_original, w, h = data.image(args.g, box)
if args.g_out:
    print("Saving input image...")
    Image.fromarray( (255*np.clip(g_original, 0, 1)).astype(np.uint8).reshape([h,w]) ).save(args.g_out)

if args.sigma > 0:
    noise = np.random.normal(loc=0.0, scale=args.sigma, size=w*h)
    g = g_original + noise
    sigma = np.sqrt(np.sum( noise**2 )/w/h)
    if args.noisy_out:
        print("Saving noisy image...")
        Image.fromarray( (255*np.clip(g, 0, 1)).astype(np.uint8).reshape([h,w]) ).save(args.noisy_out)
else:
    sigma = 0

print("***********************************")
print("Input images: ")
print(" - g = "+str(args.g))
print(" - sigma = "+str(sigma))

alpha1 = args.alpha1
alpha2 = args.alpha2
if alpha1 == 0 and alpha2 == 0:
    EI = 0
    alpha1 = EI / (EI+sigma**2)
    alpha2 = 2*sigma**2 / (EI+sigma**2)

# Create model
model = models.L1L2TVModel()
model.alpha1  = alpha1*np.ones(w*h)
model.alpha2  = alpha2*np.ones(w*h)
model.lambdaa = args.lambdaa*np.ones(w*h)
model.beta    = args.beta*np.ones(w*h)
model.gamma1  = args.gamma1*np.ones(w*h)
model.gamma2  = args.gamma2*np.ones(w*h)
print(model)

# Start timer
start_time = time.time()

algo_max_it = 30 if args.algo_max_it == 0 else args.algo_max_it
algo_epsilon = 1e-3 if args.algo_epsilon == -1 else args.algo_epsilon
algorithm = newton.L1L2TVNewtonDenoising(algo_max_it, algo_epsilon)
print(algorithm)

if args.N_refine == 0:
    # Simple Runner
    view = quadmesh.QuadMeshLeafView(w, h, w, h)
    view.create()
    view.computeIndices()
    algorithm.init(view, g, model)
    [u, p1, p2, err] = algorithm.run()
else:
    runner = runners.DenoisingCoarseToFineRunner(algorithm, w, h, args.N_refine)
    print(runner)
    runner.init(model, g, sigma)
    u = runner.run()

# stop timer
timer = time.time() - start_time

if args.out:
    print("Saving image...")
    Image.fromarray( (255*np.clip(u, 0, 1)).astype(np.uint8).reshape([h,w]) ).save(args.out)

print("Benchmark:")
AQE = np.sum( (g_original - u)**2 )/w/h
PSNR = 10*np.log10(1./AQE)
ssim_score, dif = ssim(g_original.reshape([h,w]), u.reshape([h,w]), full=True, data_range=1.)
print(" - PSNR: "+str(PSNR))
print(" - SSIM: "+str(ssim_score))
print(" - time: "+str(timer)+"s")

if args.save_benchmark:
    f = open(args.save_benchmark, "w")
    f.write("PSNR: "+str(PSNR)+"\n")
    f.write("SSIM: "+str(ssim_score)+"\n")
    f.write("time: "+str(timer)+"s")
    f.close()
print("***********************************")