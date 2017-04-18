#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//#include <omp.h>

void lacpy(char flag, double * source, double *dest, int M, int NB){
  int i, j;
  if(flag == 't'){
    for(i=0; i< NB; i++){
      memcpy(dest, source, sizeof(double)*NB);
      source += M;
      dest += NB;
    }
  }else{
    //TODO: convert from tile to LAPACK format
  }
  
}

void ge2tile(double *A, double *pA, int M, int N, int NB){
  int mt = M/NB, nt = N/NB;
  int n, m;
  double * tptr;
  double * cptr;
  for(n=0; n< nt; n++){
    for(m=0; m< mt; m++){
      
      tptr = A + n*NB*M + m*NB*NB;
      cptr = pA + n*NB*M + m*NB;
      lacpy('t', cptr, tptr, M, NB);

    }
  }
}

void dgetrf_omp(int M, int N, int NB, double *pA, int * ipiv){
  
  int i, j, k;
  int mt = M/NB, nt = N/NB; // Assume M and N are multiples of tile size
  double *A = malloc(M*N*sizeof(double));
  
  //Translate LAPACK layout to tile layout
  ge2tile(A, pA, M, N, NB);

  //#prgama omp parallel
  //#prgama omp master
  {  for(k=0; k<NB; k++){
      //#pragma omp task depend()
      //spotrf(A[k][k]);
    }
  
  }
  
  
  //Translate tile layout back to LAPACK layout
  
  
}

int main(){
  int i, j;
  int N = 8, M = 16, NB =2;
  double *pA = malloc(M*N*sizeof(double));
  double *A = malloc(M*N*sizeof(double));
  for(i=0; i< M*N; i++){
    pA[i] = i;
  }
  
  ge2tile(A, pA, M, N, NB);
  for(i=0; i< N*M/(NB*NB); i++){
    for(j=0; j< NB*NB; j++){
      printf("A %d: %f", i*NB*NB+j, A[i*NB*NB+j]);
    }
    printf("\n");
  }
  
}
