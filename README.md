# cuda-atm-solver
A bi-conjugate gradient solver for atmospheric simulations

Decompress the data
```
tar -xvzf data.tar.gz
```

Compile the solver:

```
bash compile.sh
```

Execute the solver with the data:

```
./BiCGStab.x data/matrix_cb05_10000.csr 
```

