library(dplyr)

# Read CSV files
all_imgs_results_small_model <- read.csv("all_imgs_results_small_model.csv")
all_imgs_results_big_model <- read.csv("all_imgs_results_big_model.csv")

# Select columns that start with 'ProductType_'
prdtype_cols <- grep("ProductType_", names(all_imgs_results_small_model), value = TRUE)

# Subset and sort the data frames
all_imgs_results_small_model_prdtype <- all_imgs_results_small_model %>% 
  select(c('label', prdtype_cols)) %>% 
  arrange(label) %>% 
  mutate(label_prdtype = sub("_.*", "", label))

all_imgs_results_big_model_prdtype <- all_imgs_results_big_model %>% 
  select(c('label', prdtype_cols)) %>% 
  arrange(label) %>% 
  mutate(label_prdtype = sub("_.*", "", label))

# Remove 'ProductType_' prefix from column names
names(all_imgs_results_small_model_prdtype) <- sub("ProductType_", "", names(all_imgs_results_small_model_prdtype))
names(all_imgs_results_big_model_prdtype) <- sub("ProductType_", "", names(all_imgs_results_big_model_prdtype))

# Label encoding
truelabel <- as.integer(factor(all_imgs_results_big_model_prdtype$label_prdtype))

# Get category names and create a mapping from category names to encoded labels
category_names <- unique(all_imgs_results_small_model_prdtype$label_prdtype)
category_names
category_to_encoded <- setNames(seq_along(category_names), category_names)
category_to_encoded

# Reorder columns to match the order of encoded labels
ordered_columns <- category_names[match(category_names, names(all_imgs_results_big_model_prdtype))]
ordered_columns
logitscoresA <- as.matrix(all_imgs_results_big_model_prdtype[, ordered_columns])
logitscoresB <- as.matrix(all_imgs_results_small_model_prdtype[, ordered_columns])



library(stringr)
data = read.csv("../master_list.csv")

# Replace the first part of the 'label' column with the 'ProductType' value for each row
data <- data %>%
  rowwise() %>%
  mutate(label = str_replace(label, "^[^_]*", as.character(ProductType))) %>%
  ungroup()

write.csv(data, "master_list.csv", row.names = F)