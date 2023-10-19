#include<stdio.h>
#include<iostream>
#include<math.h>
#include "src/IterativeSolver.cuh"

#define MAXCHAR 100

using namespace std;
 
int main(int argc, char **argv){

    if(argc-1 != 1) {cout<<"a.out 100K  "<<endl;return 0;}
    char*  matrix_name=argv[1];

    cout<<"Reading matrix ..."<<endl;

    IterativeSolver bicg(matrix_name, 1); 

    int num_rows = bicg.getNumRows(); 

    double*    x= new double[num_rows];
    double*    b= new double[num_rows];


    for(int i=0;i<num_rows;i++)
	{
        b[i]= 0.1*sin(i) +0.01*sin(i/2.0) +0.001*sin(i/3.0);
	x[i]=0.0;
	}
 
    bicg.setupGPU();
	
    bicg.solveGPU(b,x); 


}
