#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<iostream>
using namespace std;

class IterativeSolver{

	public:
		IterativeSolver(){};
		IterativeSolver(int, int, double*, int*, int*,int);
		IterativeSolver(string matrixName,int matrixType);
		~IterativeSolver(){};
		void setupGPU();
		void solveGPU(double*, double*);
		int getNumRows(){return nrows;};
	protected:
		// matrix data       
		double* A;
		int*    jA;
		int*    iA;
		int     nrows;
		int     nnz;

		//GPU pointers
		double* dA;
		int*    djA;
		int*    diA;
		double* dx;
		double* drhs;
		double* ddiag;

		double* aux;
		double* daux;

		int threads,blocks;

		// Auxiliary scalars
		double tolmax;
		int maxIt;
		int mattype;

		int totits;
		double totres;
		// Auxiliary vectors
		double * dr0;
		double * dr0h;
		double * dn0;
		double * dp0;
		double * dt;
		double * ds;
		double * dAx;
		double * dAx2;
		double * dy;
		double * dz;
		double * diag;

		//subroutines
		void spmv(double*, double*);
		void axpby(double*,double*,double,double);
		void yequalsx(double*, double*);
		void yequalsconst(double*, double);
		double dotxy(double*, double*);
		void zaxpbypc(double *, double* , double*, double , double );
		void multxy(double*,double*, double*);
		void zaxpby(double *, double*, double*, double , double );
		void axpy(double* , double*, double );
		void correctMatrix(double);
		void transferXandRHSToGPU(double *, double *);
		void transferSolution(double *);
};


