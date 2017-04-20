#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>


//=============================================================================

extern int  dgemm_(char *, char *, int *, int *, int *, double *, double *,
                      int *, double *, int *, double *, double *, int *);
extern void dgetrf_(int*, int*, double*, int*, int *, int*);

extern int  dtrsm_(char *, char *, char *, char *, int *, int *,
                      double *, double *, int *, double *, int *);
extern int  dsyrk_(char *, char *, int *, int *, double *, double *,
                      int *, double *, double *, int *);
//=============================================================================

void lacpy(char flag, double * source, double *dest, int M, int NB){
  int i, j;
  if(flag == 't'){
    for(i=0; i< NB; i++){
      memcpy(dest, source, sizeof(double)*NB);
      source += M;
      dest += NB;
    }
  }else if(flag == 'c'){
    for(i=0; i< NB; i++){
      memcpy(dest, source, sizeof(double)*NB);
      source += NB;
      dest += M;
    }
  }else{
    printf("oops! shouldn't have happened!\n");
    exit(-1);
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

void tile2ge(double *A, double *pA, int M, int N, int NB){
  int mt = M/NB, nt = N/NB;
  int n,m;
  double *tptr;
  double *cptr;
  for(n=0; n< nt; n++){
    for(m=0; m< mt; m++){
      
      tptr = A + n*NB*M + m*NB*NB;
      cptr = pA + n*NB*M + m*NB;
      lacpy('c', tptr, cptr, M, NB);
      
    }
  }
}

void dgetrf_omp(int M, int N, int NB, double *pA, int * ipiv){
  
  int i, j, k, m, info;
  int mt = M/NB, nt = N/NB; // Assume M and N are multiples of tile size
  double alpha = 1., neg = -1.;
  double *A = malloc(M*N*sizeof(double));
  double *akk;
  int *iptr;
  //Translate LAPACK layout to tile layout
  ge2tile(A, pA, M, N, NB);

  #pragma omp parallel
  #pragma omp master
  {
    for(k=0; k< nt; k++){ //Loop over columns
      //akk = A + k*NB*M + k*NB*NB;
      m = M - k*NB;
      //iptr = ipiv+k*NB;
      // panel
      #pragma omp task depend(inout: A[k*NB*M + k*NB*NB:m*NB])              \
                       depend(out: ipiv[k*NB:NB])                 \
                       firstprivate(k,m)
      {
        tile2ge(A+k*NB*M, pA+k*NB*M, M, NB, NB);
        
        dgetrf_(&m, &NB, pA+k*NB*M+k*NB, &M, ipiv+k*NB, &info);
        
        //TODO: update the ipiv
        
        //convert back to tile layout
        ge2tile(A+k*NB*M, pA+k*NB*M, M, NB, NB);
        
      }
      // update trailing submatrix
      for(i = k+1; i < nt; i++){
        
      }
      
    }
  }
  
  //Translate tile layout back to LAPACK layout
  tile2ge(A, pA, M, N, NB);
  
}

int main(int argc, char *argv[]){
  int i, j;
  int N = 8, M = 16, NB =4;
  double *pA = malloc(M*N*sizeof(double));
  double *A = malloc(M*N*sizeof(double));
  int * ipiv = malloc(M*sizeof(int));
  char tmp[1024], *p;
  FILE * fp;
  fp = fopen(argv[1], "r");
  
  
  for(i=0; i< M; i++){
    fgets(tmp, sizeof(tmp), fp);
    p = strtok(tmp, ",");
    for(j=0; j<N; j++){
      pA[i+j*M] = atoi(p);
      p = strtok(NULL, ",");
    }
  }

  dgetrf_omp(M, N, NB, pA, ipiv);
  
  
  for(i=0; i< M; i++){
    for(j=0; j< N; j++){
      printf("%f ", pA[i+j*M]);
    }
    printf("\n");
  }
  
  
}
