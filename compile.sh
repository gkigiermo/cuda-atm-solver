INC="-I/usr/local/cuda/include/"
INCLIB="-L/usr/local/cuda/lib64/"
nvcc -L. src/IterativeSolver.cu -c $INC $INCLIB -lcudart -lcublas -arch sm_70
nvcc -L. IterativeSolver.o main.cu -o BiCGStab.x $INC $INCLIB -lcudart -lcublas
