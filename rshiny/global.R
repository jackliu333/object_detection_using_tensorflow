library(dplyr)
library(readr)

# df = read_csv("../video_recognition/master_list.csv")
df = read_csv("../master_list.csv")
df$tag = ifelse(is.na(df$tag), "NA", df$tag)
df = df[df$tag != "test",]

all_labels = unique(df$label)
all_prdtypes = unique(df$ProductType)
all_weights = unique(df$Weight)
all_halals = unique(df$HalalStatus)
all_healths = unique(df$HealthStatus)



####### Create folder template ######
df$label2 = paste0(df$ProductType, "_", df$Weight, "_", df$HalalStatus)

# Define the vector of names
elements <- unique(df$label2)
base_dir <- file.path(getwd(), "folder_template")
# Check if the base directory exists, if not create it
if (!dir.exists(base_dir)) {
  dir.create(base_dir)
}

# Loop through the vector to create folders
for (element in elements) {
  # Construct the folder path; this example assumes the current working directory
  folder_path <- file.path(base_dir, element)
  
  # Check if the folder already exists
  if (!dir.exists(folder_path)) {
    # Create the folder if it does not exist
    dir.create(folder_path)
    cat("Created folder:", folder_path, "\n") # Print a confirmation message
  } else {
    cat("Folder already exists:", folder_path, "\n") # Print a message if the folder exists
  }
}




# length(unique(df$label2[df$tag != "test"]))
# length(unique(df$ProductType[df$tag != "test"]))
# length(unique(df$Weight[df$tag != "test"]))
# unique(df$ProductType)
