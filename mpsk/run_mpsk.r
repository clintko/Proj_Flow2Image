###
library(tidyverse)
library(MPSK)
DAT_DIR = "/data/clintko/flow_EQAPOL_normal"

###
print("read sample label")
C = read_delim(file.path(DAT_DIR, "C_1e4.txt"), delim = ",", col_names = FALSE)
C = as.matrix(C)

###
print("read Costim")
Y = read_delim(file.path(DAT_DIR, "Y_costim_1e4.txt"), delim = ",", col_names = FALSE)
Y = as.matrix(Y)

###
print("run MPSK on Costim")
set.seed(123)
Y <- scale(Y)
res <- mpsk(Y, C)

###
print("store MPSK results of Costim")
saveRDS(res, file = file.path(DAT_DIR, "mpsk_costim_1e4.RDS"))