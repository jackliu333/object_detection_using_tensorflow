library(dplyr)
library(R2jags)
library(rBayesianOptimization)
library(tictoc)
library(foreach)
library(doParallel)

# Read CSV files
all_imgs_results_bigmodel <- read.csv("../../v2/output/all_imgs_results_big_model.csv")
master_df = read.csv("../../../master_list.csv")
all_prdtypes = unique(master_df$ProductType)
all_prdtypes_bayes = colnames(all_imgs_results_df)[colnames(all_imgs_results_df) %in% all_prdtypes]

all_imgs_results_smallmodel <- read.csv("../../../small_model/new_imgs_results_small_model.csv")


######## DATA PROCESSING #######
all_imgs_results_bigmodel = all_imgs_results_bigmodel[, c("filepath", all_prdtypes_bayes, "type")]
all_imgs_results_smallmodel = all_imgs_results_smallmodel[, c("filepath", all_prdtypes_bayes, "type")]

if(length(all_prdtypes_bayes[!(all_prdtypes_bayes %in% colnames(all_imgs_results_smallmodel))]) == 0){
  print("ok")
} else{
  print("check prdtypes!")
}

total_df = 
  
  
  