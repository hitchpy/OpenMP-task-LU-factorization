#Yu Pei May 2017
#Makefile with icc and MKL for omp task LU

all: plasma getrf

plasma: test_zgetrf.c
	icc -std=c99 -I./plasma-17.1/include -L./plasma-17.1/lib test_zgetrf.c -lplasma -lcoreblas -mkl -qopenmp -o plasma

getrf: getrf_omp_task.c
	icc -std=c99 getrf_omp_task.c -mkl -qopenmp -o getrf

clean:
	rm getrf plasma 
