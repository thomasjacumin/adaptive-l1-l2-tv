#!/bin/bash

run__help() {
  echo "help"
}

run__download() {
    mkdir data
    rm -Rf data/optical_flow
    wget https://vision.middlebury.edu/flow/data/comp/zip/other-gray-twoframes.zip
    unzip -qq other-gray-twoframes.zip "other-data-gray/**" -d data/optical_flow/
    rm other-gray-twoframes.zip
    wget https://vision.middlebury.edu/flow/data/comp/zip/other-gt-flow.zip
    unzip -qq other-gt-flow.zip "other-gt-flow/**" -d data/optical_flow/
    rm other-gt-flow.zip
}

run__installpipdependencies() {
    pip install matplotlib scipy scikit-image alive-progress pillow
}

run__restart() {
    rm -Rf results
    run
}

run() {
    mkdir results
    run__denoising
    run__opticalflow
}

run__denoising() {
    mkdir results/denoising
    for dir in data/optical_flow/other-data-gray/*; do
        if [ -d "$dir" ]; then
            input=${dir##*/}
            g=data/optical_flow/other-data-gray/$input/frame10.png
            ground_truth=data/optical_flow/other-gt-flow/$input/flow10.flo
    
            if test -f "$ground_truth"; then
                mkdir results/denoising/$input
    
                # sigma = 0.1
                ## Uniform
                if [ ! -f "results/denoising/$input/.out.0.1.unif.sucess" ]; then
                    python3 denoising.py $g --noisy-out=results/denoising/$input/noisy.0.1.png --out=results/denoising/$input/unif.0.1.png --save-benchmark=results/denoising/$input/unif.0.1.benchmark.txt --sigma=0.1 \
                        --beta=0 --lambdaa=1 --alpha2=10 --gamma1=2e-4 --gamma2=2e-4 \
                        --algo-max-it=100
                    touch results/denoising/$input/.out.0.1.unif.sucess
                fi
                ## Coarse-to-fine
                if [ ! -f "results/denoising/$input/.out.0.1.ctf.sucess" ]; then
                    python3 denoising.py $g --noisy-out=results/denoising/$input/noisy.0.1.png --out=results/denoising/$input/ctf.0.1.png --save-benchmark=results/denoising/$input/ctf.0.1.benchmark.txt --sigma=0.1 \
                        --beta=0 --lambdaa=1 --alpha2=10 --gamma1=2e-4 --gamma2=2e-4 \
                        --N-refine=6 --algo-max-it=100
                    touch results/denoising/$input/.out.0.1.ctf.sucess
                fi
    
                # sigma = 0.05
                ## Uniform
                if [ ! -f "results/denoising/$input/.out.0.05.unif.sucess" ]; then
                    python3 denoising.py $g --noisy-out=results/denoising/$input/noisy.0.05.png --out=results/denoising/$input/unif.0.05.png --save-benchmark=results/denoising/$input/unif.0.05.benchmark.txt --sigma=0.05 \
                        --beta=0 --lambdaa=1 --alpha2=10 --gamma1=2e-4 --gamma2=2e-4 \
                        --algo-max-it=100
                    touch results/denoising/$input/.out.0.05.unif.sucess
                fi
                ## Coarse-to-fine
                if [ ! -f "results/denoising/$input/.out.0.05.ctf.sucess" ]; then
                    python3 denoising.py $g --noisy-out=results/denoising/$input/noisy.0.05.png --out=results/denoising/$input/ctf.0.05.png --save-benchmark=results/denoising/$input/ctf.0.05.benchmark.txt --sigma=0.05 \
                        --beta=0 --lambdaa=1 --alpha2=10 --gamma1=2e-4 --gamma2=2e-4 \
                        --N-refine=6 --algo-max-it=100
                    touch results/denoising/$input/.out.0.05.ctf.sucess
                fi
    
                # sigma = 0.01
                ## Uniform
                if [ ! -f "results/denoising/$input/.out.0.01.unif.sucess" ]; then
                    python3 denoising.py $g --noisy-out=results/denoising/$input/noisy.0.01.png --out=results/denoising/$input/unif.0.01.png --save-benchmark=results/denoising/$input/unif.0.01.benchmark.txt --sigma=0.01 \
                        --beta=0 --lambdaa=1 --alpha2=10 --gamma1=2e-4 --gamma2=2e-4 \
                        --algo-max-it=100
                    touch results/denoising/$input/.out.0.01.unif.sucess
                fi
                ## Coarse-to-fine
                if [ ! -f "results/denoising/$input/.out.0.01.ctf.sucess" ]; then
                    python3 denoising.py $g --noisy-out=results/denoising/$input/noisy.0.01.png --out=results/denoising/$input/ctf.0.01.png --save-benchmark=results/denoising/$input/ctf.0.01.benchmark.txt --sigma=0.01 \
                        --beta=0 --lambdaa=1 --alpha2=10 --gamma1=2e-4 --gamma2=2e-4 \
                        --N-refine=6 --algo-max-it=100
                    touch results/denoising/$input/.out.0.01.ctf.sucess
                fi
            fi
        fi
    done
}

run__opticalflow() {
    mkdir results/optical-flow
    for dir in data/optical_flow/other-data-gray/*; do
        if [ -d "$dir" ]; then
            input=${dir##*/}
            frame10=data/optical_flow/other-data-gray/$input/frame10.png
            frame11=data/optical_flow/other-data-gray/$input/frame11.png
            ground_truth=data/optical_flow/other-gt-flow/$input/flow10.flo

            if test -f "$ground_truth"; then
                mkdir results/optical-flow/$input
                
                # Generate ground truth           
                normalizing=$(./bin/color_flow $ground_truth results/$input/flow10.png | grep -Eo '^max motion: [[:digit:]]+([.][[:digit:]]+)?' | grep -Eo '[[:digit:]]+([.][[:digit:]]+)?$')
                ./bin/color_flow $ground_truth results/optical-flow/$input/flow10.png $normalizing
                echo "optical flow will be normalize by ${normalizing}"
    
                # L1
                if [ ! -f "results/optical-flow/$input/.out.l1.sucess" ]; then
                    python3 optical_flow.py $frame10 $frame11 --ground-truth=$ground_truth --out=results/optical-flow/$input/l1.flo --normalize --save-benchmark=results/optical-flow/$input/l1.benchmark.txt \
                    --beta=1e-5 --alpha1=3 --lambdaa=1 --gamma1=2e-4 --gamma2=2e-4 \
                    --algo=newton --algo-epsilon=1e-3 --algo-max-it=100 --algo-theta=1
                    ./bin/color_flow results/optical-flow/$input/l1.flo results/optical-flow/$input/l1.png $normalizing
                    touch results/optical-flow/$input/.out.l1.sucess
                fi

                # warping
                if [ ! -f "results/optical-flow/$input/.out.warp.sucess" ]; then
                    python3 optical_flow.py $frame10 $frame11 --ground-truth=$ground_truth --out=results/optical-flow/$input/warp.flo --normalize --save-benchmark=results/optical-flow/$input/warp.benchmark.txt \
                    --beta=1e-5 --alpha1=3 --lambdaa=1 --gamma1=2e-4 --gamma2=2e-4 \
                    --algo=newton --algo-epsilon=1e-3 --algo-max-it=100 --algo-theta=1 \
                    --warp --warp-epsilon=1e-2 --warp-max-it=100
                    ./bin/color_flow results/optical-flow/$input/warp.flo results/optical-flow/$input/warp.png $normalizing
                    touch results/optical-flow/$input/.out.warp.sucess
                fi

                # warping while c-t-f
                if [ ! -f "results/optical-flow/$input/.out.ours.sucess" ]; then
                    python3 optical_flow.py $frame10 $frame11 --ground-truth=$ground_truth --out=results/optical-flow/$input/ours.flo --normalize --save-benchmark=results/optical-flow/$input/ours.benchmark.txt \
                        --beta=1e-5 --alpha1=3 --lambdaa=1 --gamma1=2e-4 --gamma2=2e-4 \
                        --algo=newton --algo-epsilon=1e-3 --algo-max-it=100 --algo-theta=1 \
                        --N-refine=6 --ctf-epsilon=1e-2 --ctf-warp --ctf-mark=0.75
                    ./bin/color_flow results/optical-flow/$input/ours.flo results/optical-flow/$input/ours.png $normalizing
                    touch results/optical-flow/$input/.out.ours.sucess
                fi
            fi
        fi
    done
}

if [ "$1" = "help" ]; then
    run__help
elif [ "$1" = "download" ]; then
    run__download
elif [ "$1" = "install" ]; then
    run__installpipdependencies
elif [ "$1" = "restart" ]; then
    run__restart
elif [ "$1" = "denoising" ]; then
    run__denoising
elif [ "$1" = "opticalflow" ]; then
    run__opticalflow
else
    run
fi
