library(shiny)
library(shinydashboard)
library(rsconnect)
library(tidyverse)
library(reticulate)
library(purrr)
library(stringr)

# Create conda env if not exist
if(!("foodbank_py37" %in% conda_list()$name)){
  # conda_create("foodbank_py37", python_version = "3.7")
  use_condaenv("foodbank_py37", required = TRUE)
  # Set up python libraries for object detection
  source_python("setup.py")
}


# Check python version
# py_config()

# Load OD model and prediction functions
source_python("image_classification.py")

# Function to convert scoring result to a df
convert_scoring_rst <- function(df){
  all_df = NULL
  for(x in names(df)){
    tmp_name = x
    tmp_val = df[[x]]
    tmp_val = as.numeric(as.character(tmp_val))
    tmp_row = data.frame(tmp_name, tmp_val)
    all_df = rbind(all_df, tmp_row)
  }
  colnames(all_df) = c("Category", "Prediction")
  
  return(all_df)
}

# Load all image names from current directory
ALL_IMGS = list.files("images/")
ALL_CATS = unique(unlist(purrr::map(ALL_IMGS, function(x) str_split(x,"_")[[1]][1])))
