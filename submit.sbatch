#!/bin/bash
#SBATCH --job-name="openmp_task_LU"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --export=ALL
#SBATCH -t 01:20:00

#Set the number of openmp threads
#export OMP_NUM_THREADS=24

#for i in 400 800 1600 4000 8000 12000 16000 20000 24000 26000 30000 32000
#do
#  for j in 1 2 3
#    do
#    ./getrf $i
#    ./plasma $i
#    done
#done

for i in 8 12 16 20 24
do 
  for j in 1 
    do
    export OMP_NUM_THREADS=$i
    ./getrf 30000
    done
done
