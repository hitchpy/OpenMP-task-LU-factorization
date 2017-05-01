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
extern void cblas_dswap(int  , double *, int , double *, int);
extern void cblas_daxpy(int , double , double *x, int , double *, int);
extern double cblas_dnrm2(const int, const double *, const int);
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

void ge2tile(double *A, double *pA, int M, int N, int NB, int LDA){
  int mt = M/NB, nt = N/NB;
  int n, m;
  double * tptr;
  double * cptr;
  for(n=0; n< nt; n++){
    for(m=0; m< mt; m++){
      
      tptr = A + n*NB*LDA + m*NB*NB;
      cptr = pA + n*NB*LDA + m*NB;
      lacpy('t', cptr, tptr, LDA, NB);

    }
  }
}

void tile2ge(double *A, double *pA, int M, int N, int NB, int LDA){
  int mt = M/NB, nt = N/NB;
  int n,m;
  double *tptr;
  double *cptr;
  for(n=0; n< nt; n++){
    for(m=0; m< mt; m++){
      
      tptr = A + n*NB*LDA + m*NB*NB;
      cptr = pA + n*NB*LDA + m*NB;
      lacpy('c', tptr, cptr, LDA, NB);
      
    }
  }
}

void core_zgeswp(double *A, int NB, int k1, int k2, int *ipiv){
  int m, m1, m2;
  for(m= k1; m<k2; m++){
    if(ipiv[m]-1 != m){
      m1 = m;
      m2 = ipiv[m] - 1;
      
      cblas_dswap(NB, A+(m1/NB)*(NB*NB)+m1%NB, NB,
                      A+(m2/NB)*(NB*NB)+m2%NB, NB);
    }
  }
}

void dgetrf_omp(int M, int N, int NB, double *pA, int * ipiv){
  
  int info;
  int mt = M/NB, nt = N/NB; // Assume M and N are multiples of tile size
  double alpha = 1., neg = -1.;
  double *A = malloc(M*N*sizeof(double));
  double *akk, *akj;
  int *iptr;
  double time1, time2, elapsed;
  //Translate LAPACK layout to tile layout
  ge2tile(A, pA, M, N, NB, M);
  
  time1 = omp_get_wtime();
  #pragma omp parallel
  #pragma omp master
  {
    for(int k=0; k< nt; k++){ //Loop over columns
      int m = M - k*NB;
      akk = A + k*NB*M + k*NB*NB;
      // panel
      #pragma omp task depend(inout: A[k*NB*M:M*NB])    \
                       depend(out: ipiv[k*NB:NB])    \
                       firstprivate(akk, m)
      {
        tile2ge(A+k*NB*M+k*NB*NB, pA+k*NB*M+k*NB, m, NB, NB, M);
        
        dgetrf_(&m, &NB, pA+k*NB*M+k*NB, &M, ipiv+k*NB, &info);
        
        //update the ipiv
        for(int i=k*NB; i< k*NB+NB; i++){
          ipiv[i] += k*NB;
        }
        
        //convert back to tile layout
        ge2tile(A+k*NB*M+k*NB*NB, pA+k*NB*M+k*NB, m, NB, NB, M);
        
      }
      
      // update trailing submatrix
      for(int j = k+1; j < nt; j++){
        akj = A+j*NB*M+k*NB*NB; // TRSM and submatrix in this column
        
        #pragma omp task depend(in: A[k*NB*M:M*NB])              \
                         depend(in: ipiv[k*NB:NB])               \
                         depend(inout: A[j*NB*M: M*NB])             \
                         firstprivate(akk, akj, m)
        {
          // laswp
          int k1 = k*NB;
          int k2 = k*NB+NB;
          core_zgeswp(A+j*NB*M, NB, k1, k2, ipiv);
          
          // trsm
          dtrsm_("l", "l", "n", "u", &NB, &NB,
                 &alpha, akk, &NB, akj, &NB);
          
          // gemm
          for(int i= k+1; i< mt; i++){
            dgemm_("n", "n", &NB, &NB, &NB,
                   &neg, A+k*NB*M + i*NB*NB,
                   &NB, akj, &NB,
                   &alpha, A+j*NB*M + i*NB*NB, &NB);
          }
          
        }
      
      }
      
    }
    
    // pivoting to the left.
    for(int t =1; t < nt; t++){
      int k1 = t*NB;
      int k2 = N; // Assume M >= N
      
      #pragma omp task depend(in:ipiv[(nt-1)*NB:NB]) \
                       depend(inout:A[t*NB*M:M*NB])
      {
        core_zgeswp(A+(t-1)*NB*M, NB, t*NB, N, ipiv);
      }
    }
  } // End of parallel region
  
  
  
  time2 = omp_get_wtime();
  elapsed = time2 - time1;
  
  //Translate tile layout back to LAPACK layout
  tile2ge(A, pA, M, N, NB, M);
  
  printf("My time is %f, Flops is %f\n", elapsed, 2.*N*N*N/(3.*elapsed*1e9));
}

int main(int argc, char *argv[]){
  int i, j, info;
  int N = atoi(argv[1]), M = atoi(argv[1]), NB =200;
  double neg = -1.0;
  double *pA = malloc(M*N*sizeof(double));
  double *pB = malloc(M*N*sizeof(double));
  double *A = malloc(M*N*sizeof(double));
  int * ipiv = malloc(M*sizeof(int));
  char tmp[1024], *p;
  FILE * fp;
  double time1, time2, elapsed;
  
  
  //Generate random data
  
  for(i=0; i< N; i++){
    for(j=0; j< M; j++)
      pA[j+i*N] = pB[j+i*N] = ((double)rand())/RAND_MAX*10.;
  }
  
  for(i=0; i<N; i++){
    ipiv[i] = i;
  }
  
  //Calling my LU decomposition
  
  dgetrf_omp(M, N, NB, pA, ipiv);
  
  //Result validation
  
  time1 = omp_get_wtime();
  dgetrf_(&M, &N, pB, &M, ipiv, &info);
  time2 = omp_get_wtime();
  elapsed = time2 - time1;
  
  printf("LAPACK time is %f, Flops is %f\n", elapsed, 2.*N*N*N/(3.*elapsed*1e9));
  
  cblas_daxpy(M*N, neg, pA, 1, pB, 1);
  printf("Error: %e\n", cblas_dnrm2(M*N, pB, 1));
  
  for(i=0; i< 10; i++){
    for(j=0; j< 10; j++){
      printf("%f ", pB[i+j*M]);
    }
    printf("\n");
  }
  for(i=0; i< 10; i++){
    for(j=0; j< 10; j++){
      printf("%f ", pA[i+j*M]);
    }
    printf("\n");
  }
}
