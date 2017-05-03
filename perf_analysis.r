setwd('~/Documents/Courses/UTK/COSC594_Scientific_Computing/final_project/')
library(ggplot2)
library(reshape2)

dat = scan('slurm-9109486.out')
mat = matrix(dat, ncol=3, byrow = T)

dat0 = scan('slurm-9109904.out')
dat = scan('slurm-9111090.out')
threads = matrix(dat0, ncol=2, byrow = T)
temp = matrix(dat, ncol=2, byrow = T)
threads = rbind(threads, temp)
threads = threads[c(1,4,7,9:13),]

mat = data.frame(mat)
threads = data.frame(threads)
threads$threads = c(1,2,4,8,12,16,20,24)
colnames(threads) = c('MyLU', 'LAPACK', 'Num_Threads')

mat = mat[-c(1:6),]
mat$Size = rep(c(1600, 4000, 8000,
             12000, 16000, 20000,
             24000, 26000, 30000, 32000), each=3)
temp = split(mat, mat$Size)
mat = t(sapply(temp, apply, 2, mean))
mat = data.frame(mat)
row.names(mat) = NULL
colnames(mat) = c('MyLU', 'LAPACK', 'PLASMA', 'Size')

mat = melt(mat, id.vars='Size', variable.name = 'Package',
           value.name = 'GFlops')

ggplot(mat,aes(x=Size, y= GFlops, colour =Package)) + 
      geom_line() + geom_point()

threads = melt(threads, id.vars='Num_Threads', variable.name = 'Package',
               value.name = 'GFlops')

ggplot(threads, aes(x=Num_Threads, y= GFlops, colour =Package)) + 
  geom_line() + geom_point()
