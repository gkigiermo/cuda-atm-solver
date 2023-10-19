#include<cuda.h>
#include<cublas.h>
#include "IterativeSolver.cuh"

IterativeSolver::IterativeSolver(int tnrows,int tnnz,double *tA, int *tjA, int *tiA, int tmattype)
{
	nrows=tnrows;
	nnz=tnnz;
	mattype=tmattype;
	A=tA;
	jA=tjA;
	iA=tiA;

	threads=128;
	blocks=(nrows+threads-1)/threads;

	cudaMalloc((void**)&dA, nnz*sizeof(double));
	
	cudaMalloc((void**)&djA, nnz*sizeof(int));
	cudaMalloc((void**)&diA, (nrows+1)*sizeof(int));

	cudaMalloc((void**)&dx, nrows*sizeof(double));
	cudaMalloc((void**)&drhs,nrows*sizeof(double));
	
	cudaMemcpy(dA,A,nnz*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(djA,jA,nnz*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(diA,iA,(nrows+1)*sizeof(int),cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dr0, nrows*sizeof(double));
	cudaMalloc((void**)&dr0h,nrows*sizeof(double));
	cudaMalloc((void**)&dn0,nrows*sizeof(double));
	cudaMalloc((void**)&dp0, nrows*sizeof(double));
	cudaMalloc((void**)&dt, nrows*sizeof(double));
	cudaMalloc((void**)&ds, nrows*sizeof(double));
	cudaMalloc((void**)&dAx, nrows*sizeof(double));
	cudaMalloc((void**)&dAx2, nrows*sizeof(double));
	cudaMalloc((void**)&dy, nrows*sizeof(double));
	cudaMalloc((void**)&dz, nrows*sizeof(double));
	cudaMalloc((void**)&ddiag, nrows*sizeof(double));

	cudaMalloc((void**)&daux,nrows*sizeof(double));

	aux=(double*)malloc(sizeof(double)*blocks);
	diag=(double*)malloc(sizeof(double)*nrows);

	maxIt=500;
	tolmax=1e-6;
}

IterativeSolver::IterativeSolver(string matrixName, int tmattype)
{
	FILE *fp;
	
	printf("Matrix name: %s \n", matrixName.c_str());
	fp= fopen(matrixName.c_str(),"r");

	int sizes[5];
	for(int i=0;i<5;i++)
		fscanf(fp," %d",&sizes[i]);
	fscanf(fp," \n");

	nnz=sizes[0];
	nrows=sizes[2];
	printf("Non-zero entries: %d, rows: %d \n", nnz, nrows);
	A=new double[nnz];
	jA=new int[nnz];
	iA= new int[nrows+1];

	for(int i=0;i<nnz;i++)
		fscanf(fp," %lf",&A[i]);
	fscanf(fp," \n");
	for(int i=0;i<nnz;i++)
		fscanf(fp," %d",&jA[i]);
	fscanf(fp," \n");
	for(int i=0;i<nrows+1;i++)
		fscanf(fp," %d",&iA[i]);
	fscanf(fp," \n");

	fclose(fp);
	threads=128;
	blocks=(nrows+threads-1)/threads;

	cudaMalloc((void**)&dA, nnz*sizeof(double));
	
	cudaMalloc((void**)&djA, nnz*sizeof(int));
	cudaMalloc((void**)&diA, (nrows+1)*sizeof(int));

	cudaMalloc((void**)&dx, nrows*sizeof(double));
	cudaMalloc((void**)&drhs,nrows*sizeof(double));
	
	cudaMalloc((void**)&dr0, nrows*sizeof(double));
	cudaMalloc((void**)&dr0h,nrows*sizeof(double));
	cudaMalloc((void**)&dn0,nrows*sizeof(double));
	cudaMalloc((void**)&dp0, nrows*sizeof(double));
	cudaMalloc((void**)&dt, nrows*sizeof(double));
	cudaMalloc((void**)&ds, nrows*sizeof(double));
	cudaMalloc((void**)&dAx, nrows*sizeof(double));
	cudaMalloc((void**)&dAx2, nrows*sizeof(double));
	cudaMalloc((void**)&dy, nrows*sizeof(double));
	cudaMalloc((void**)&dz, nrows*sizeof(double));
	cudaMalloc((void**)&ddiag, nrows*sizeof(double));

	cudaMalloc((void**)&daux,nrows*sizeof(double));

	cudaMemcpy(dA,A,nnz*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(djA,jA,nnz*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(diA,iA,(nrows+1)*sizeof(int),cudaMemcpyHostToDevice);

	aux=(double*)malloc(sizeof(double)*blocks);

	diag=(double*)malloc(sizeof(double)*nrows);

	maxIt=500;
	tolmax=1e-6;
}


void IterativeSolver::correctMatrix(double alpha)
{
	for(int row=0;row<nrows;row++){
		int istart=iA[row];
		int iend  =iA[row+1];
		for(int i=istart;i<iend;i++){
			if(jA[i]==row){
				A[i]=1.0 + alpha*A[i];	
			}
			else
			{
				A[i]= alpha*A[i];
			}		
		}
	}

}

void IterativeSolver::setupGPU()
{
	correctMatrix(-6.0);

	//Diagonal precond
	for(int row=0;row<nrows;row++){
		int istart=iA[row];
		int iend  =iA[row+1];
		for(int i=istart;i<iend;i++){
			if(jA[i]==row){
				if(A[i]!=0.0)
					diag[row]= 1.0/A[i];
				else
					diag[row]= 1.0;
			}			
		}
	}

	cudaMemcpy(dA,A,nnz*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(ddiag,diag,nrows*sizeof(double),cudaMemcpyHostToDevice);

}

// y= a*x+ b*y
__global__ void cudaaxpby(double* dy,double* dx, double a, double b, int nrows)
{
	int row= threadIdx.x + blockDim.x*blockIdx.x;
   	if(row < nrows){
		dy[row]= a*dx[row] + b*dy[row];
	}
}

void IterativeSolver::axpby(double* y, double*x, double a, double b)
{
	dim3 dimGrid(blocks,1,1);
	dim3 dimBlock(threads,1,1); 
	cudaaxpby<<<dimGrid,dimBlock>>>(y,x,a,b,nrows);

}

// y = x
__global__ void cudayequalsx(double* dy,double* dx,int nrows)
{
	int row= threadIdx.x + blockDim.x*blockIdx.x;
   	if(row < nrows){
		dy[row]=dx[row];
	}
}

void IterativeSolver::yequalsx(double* y, double* x)
{
	dim3 dimGrid(blocks,1,1);
	dim3 dimBlock(threads,1,1); 
	cudayequalsx<<<dimGrid,dimBlock>>>(y,x,nrows);

}
// y = constant
__global__ void cudasetconst(double* dy,double constant,int nrows)
{
	int row= threadIdx.x + blockDim.x*blockIdx.x;
   	if(row < nrows){
		dy[row]=constant;
	}
}

void IterativeSolver::yequalsconst(double* y, double constant)
{
	dim3 dimGrid(blocks,1,1);
	dim3 dimBlock(threads,1,1); 
	cudasetconst<<<dimGrid,dimBlock>>>(y,constant,nrows);
}


__global__ void cudadot(double *g_idata1, double *g_idata2, double *g_odata, unsigned int n)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    double mySum = (i < n) ? g_idata1[i]*g_idata2[i] : 0;


    if (i + blockDim.x < n)
        mySum += g_idata1[i+blockDim.x]*g_idata2[i+blockDim.x];
    
    sdata[tid] = mySum;
    __syncthreads();

    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
            sdata[tid] = mySum = mySum + sdata[tid + s];

        __syncthreads();
    }

    if (tid == 0)g_odata[blockIdx.x] = sdata[0];
}

__global__ void finalred( double *g_odata, unsigned int n)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    double mySum =  (tid < n) ? g_odata[tid] : 0;

    sdata[threadIdx.x] = mySum;
    __syncthreads();

    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
            sdata[tid] = mySum = mySum + sdata[tid + s];

        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}
double IterativeSolver::dotxy(double* x, double* y)
{
	int nblocks =(blocks+1)/2;
	dim3 dimGrid(nblocks,1,1);
	dim3 dimBlock(threads,1,1);
	cudadot<<<dimGrid,dimBlock,threads*sizeof(double)>>>(x,y,daux,nrows);
	int redsize= sqrt(nblocks) +1;
	redsize=pow(2,redsize);

	dim3 dimGrid2(1,1,1);
	dim3 dimBlock2(redsize,1,1);

	double dot;
	finalred<<<dimGrid2,dimBlock2,redsize*sizeof(double)>>>(daux,nblocks);

	cudaMemcpy(&dot, daux, sizeof(double), cudaMemcpyDeviceToHost);

	return dot;
}

// z= a*z + x + b*y
__global__ void cudazaxpbypc(double* dz, double* dx,double* dy, double a, double b, int nrows)
{
	int row= threadIdx.x + blockDim.x*blockIdx.x;
   	if(row < nrows){
		dz[row]=a*dz[row]  + dx[row] + b*dy[row];
	}
}

void IterativeSolver::zaxpbypc(double *z, double* x, double*y, double a, double b)
{
	dim3 dimGrid(blocks,1,1);
	dim3 dimBlock(threads,1,1); 
	cudazaxpbypc<<<dimGrid,dimBlock>>>(z,x,y,a,b,nrows);
}

// z= x*y
__global__ void cudamultxy(double* dz, double* dx,double* dy, int nrows)
{
	int row= threadIdx.x + blockDim.x*blockIdx.x;
   	if(row < nrows){
		dz[row]=dx[row]*dy[row];
	}
}

void IterativeSolver::multxy(double* z,double* x, double*y)
{
     	dim3 dimGrid(blocks,1,1);
	dim3 dimBlock(threads,1,1); 
   
	cudamultxy<<<dimGrid,dimBlock>>>(z,x,y,nrows);
}

// z= a*x + b*y
__global__ void cudazaxpby(double* dz, double* dx,double* dy, double a, double b, int nrows)
{
	int row= threadIdx.x + blockDim.x*blockIdx.x;
   	if(row < nrows){
		dz[row]=a*dx[row] + b*dy[row];
	}
}


void IterativeSolver::zaxpby(double *z, double* x, double*y, double a, double b)
{
     	dim3 dimGrid(blocks,1,1);
	dim3 dimBlock(threads,1,1); 
   
	cudazaxpby<<<dimGrid,dimBlock>>>(z,x,y,a,b,nrows);
}

// y= a*x + y
__global__ void cudaaxpy(double* dy,double* dx, double a, int nrows)
{
	int row= threadIdx.x + blockDim.x*blockIdx.x;
   	if(row < nrows){
		dy[row]=a*dx[row] + dy[row];
	}
}

void IterativeSolver::axpy(double* y, double*x, double a)
{
     	dim3 dimGrid(blocks,1,1);
	dim3 dimBlock(threads,1,1); 
   
	cudaaxpy<<<dimGrid,dimBlock>>>(y,x,a,nrows);
}

// x=A*b
__global__ void cudaSpmvCSR(double* dx, double* db, int nrows, double* dA, int* djA, int* diA)
{
	int row= threadIdx.x + blockDim.x*blockIdx.x;
   	if(row < nrows){
            int jstart = diA[row];
            int jend   = diA[row+1];
            double sum = 0.0;
            for(int j=jstart; j<jend; j++)
		{
			sum+= db[djA[j]]*dA[j];
		}
	    dx[row]=sum;
	}
 
}

__global__ void cudaSpmvCSC(double* dx, double* db, int nrows, double* dA, int* djA, int* diA)
{
	double mult;
	int row= threadIdx.x + blockDim.x*blockIdx.x;
   	if(row < nrows){
            int jstart = diA[row];
            int jend   = diA[row+1];
            for(int j=jstart; j<jend; j++)
		{
			mult = db[row]*dA[j];
			atomicAdd(&(dx[djA[j]]),mult);
		}
	}
}


void IterativeSolver::spmv(double* x, double* b)
{
	dim3 dimGrid(blocks,1,1);
	dim3 dimBlock(threads,1,1); 

	if(mattype==0) {
		cudaSpmvCSR<<<dimGrid,dimBlock>>>(x, b, nrows, dA, djA, diA);		
	}
	else
	{
		cudasetconst<<<dimGrid,dimBlock>>>(x,0.0, nrows);
		cudaSpmvCSC<<<dimGrid,dimBlock>>>(x, b, nrows, dA, djA, diA);
	}
}
void IterativeSolver::transferXandRHSToGPU(double *rhs, double *x)
{
	cudaMemcpy(drhs,rhs,nrows*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dx,x,nrows*sizeof(double),cudaMemcpyHostToDevice);
}

void IterativeSolver::transferSolution(double *x)
{
	cudaMemcpy(x,dx,nrows*sizeof(double),cudaMemcpyDeviceToHost);
}

void IterativeSolver::solveGPU(double *rhs, double *x)
{
	double alpha,rho0,omega0,beta,rho1,temp1,temp2;

	transferXandRHSToGPU(rhs,x);

	spmv(dr0,dx);

	axpby(dr0,drhs,1.0,-1.0); 

	yequalsx(dr0h,dr0);

	alpha  = 1.0;
	rho0   = 1.0;
	omega0 = 1.0;

	yequalsconst(dn0,0.0); 
	yequalsconst(dp0,0.0); 


	for(int it=0;it<maxIt;it++){

		rho1=dotxy(dr0,dr0h); 

		beta=(rho1/rho0)*(alpha/omega0);

		zaxpbypc(dp0,dr0,dn0,beta,-1.0*omega0*beta);

		multxy(dy,ddiag,dp0);

		spmv(dn0,dy);

		temp1=dotxy(dr0h,dn0);

		alpha=rho1/temp1;

		zaxpby(ds,dr0,dn0,1.0,-1.0*alpha);

		multxy(dz,ddiag,ds);

		spmv(dt,dz);

		multxy(dAx2,ddiag,dt);

		temp1=dotxy(dz,dAx2);

		temp2=dotxy(dAx2,dAx2);

		omega0= temp1/temp2;

		axpy(dx,dy,alpha); 

		axpy(dx,dz,omega0);

		zaxpby(dr0,ds,dt,1.0,-1.0*omega0);

		temp1=dotxy(dr0,dr0);
		temp1=sqrt(temp1);

		printf("Iteration:%d  residual: %e\n", it, temp1);

		if(temp1<tolmax){
			totits=it;
			totres=temp1;
			break;
		}

		rho0=rho1;

	}

	transferSolution(x);
}


