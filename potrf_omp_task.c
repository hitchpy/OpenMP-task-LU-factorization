#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>


//=============================================================================

extern int  dgemm_(char *, char *, int *, int *, int *, double *, double *,
                      int *, double *, int *, double *, double *, int *);
extern void dpotrf_(char*, int*, double*, int*, int *);

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
  
  int i, j, k, info;
  int mt = M/NB, nt = N/NB; // Assume M and N are multiples of tile size
  double alpha = 1., neg = -1.;
  double *A = malloc(M*N*sizeof(double));
  double *akk, *aki, *akj, *aii, *aji;
  //Translate LAPACK layout to tile layout
  ge2tile(A, pA, M, N, NB);

  #pragma omp parallel
  #pragma omp master
  {
    for(k=0; k<nt; k++){
      akk = A + k*NB*M + k*NB*NB;
        #pragma omp task depend(inout: akk[0:NB*NB]) firstprivate(akk)
      {
        dpotrf_("U", &NB, akk, &NB, &info);
      }
      for(i= k+1; i< nt; i++){
        aki = A+i*NB*M + k*NB*NB;
        #pragma omp task depend(in: akk[0:NB*NB])    \
                         depend(inout: aki[0:NB*NB]) \
                         firstprivate(akk, aki)
        {
          dtrsm_("l", "u", "t", "n", &NB, &NB, &alpha, akk,
                 &NB, aki, &NB);
        }
      }
      
      //Update trailing submatrix
      for(i= k+1; i<nt; i++){
        aki = A+i*NB*M + k*NB*NB;
        for(j= k+1; j<i; j++){
          akj = A+j*NB*M + k*NB*NB;
          aji = A+i*NB*M + j*NB*NB;
          #pragma omp task depend(in: aki[0:NB*NB])   \
                           depend(in: akj[0:NB*NB])   \
                           depend(inout: aji[0:NB*NB])\
                           firstprivate(aki, akj, aji)
          {
            dgemm_("t", "n", &NB, &NB, &NB, &neg, akj,
                  &NB, aki, &NB, &alpha,aji, &NB);

          }
        }
        aii = A+i*NB*M + i*NB*NB;
        #pragma omp task depend(in: aki[0:NB*NB])    \
                         depend(inout: aii[0:NB*NB]) \
                         firstprivate(aki, aii)
        {
          dsyrk_("u", "t", &NB, &NB, &neg, aki, &NB,
                 &alpha, aii, &NB);
        }
      }
    }
  }
  
  //Translate tile layout back to LAPACK layout
  tile2ge(A, pA, M, N, NB);
  
}

int main(){
  int i, j, info;
  int N = 8, M = 8, NB =2;
  double *pA = malloc(M*N*sizeof(double));
  double *pB = malloc(M*N*sizeof(double));
  double *A = malloc(M*N*sizeof(double));
  int * ipiv;
  
  for(i=0; i< M; i++){
    pA[i+i*M] = pB[i+i*M] = M+i;
  }
  for(i=0; i< M-1; i++){
    pA[i+(i+1)*M] = pB[i+(i+1)*M] = 0.5 + 1./(i+1);
  }
  for(i=0; i< M-1; i++){
    pA[i+1+i*M] = pB[i+1+i*M] = 0.5 + 1./(i+1);
  }
  //ge2tile(A, pA, M, N, NB);
  for(i=0; i< M; i++){
    for(j=0; j< N; j++){
      printf("%f ", pA[i+j*M]);
    }
    printf("\n");
  }
  printf("Solution\n");
  dgetrf_omp(M, N, NB, pA, ipiv);
  
  //memset(pA,0, sizeof(double)*M*N);
  //tile2ge(A, pA, M, N, NB);
  
  
  for(i=0; i< M; i++){
    for(j=0; j< N; j++){
      printf("%f ", pA[i+j*M]);
    }
    printf("\n");
  }
  printf("LAPACK solution\n");
  dpotrf_("U", &N, pB, &N, &info);
  for(i=0; i< M; i++){
    for(j=0; j< N; j++){
      printf("%f ", pB[i+j*M]);
    }
    printf("\n");
  }
  
}
