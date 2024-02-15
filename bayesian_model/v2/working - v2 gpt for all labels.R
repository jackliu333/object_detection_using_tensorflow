library(dplyr)
library(R2jags)
library(purrr)

# Read CSV files
all_imgs_results_gpt <- read.csv("GPT_model/chatgpt_prediction_v2.csv")
all_imgs_results_big_model <- read.csv("NN_model/regularized/all_imgs_results_big_model.csv")
product_group_df = read.csv("../product_group.csv")

######## DATA PROCESSING #######
# add group to gpt predictions 
all_imgs_results_gpt2 = all_imgs_results_gpt %>% 
  rename(filepath = img_filename) %>% 
  left_join(product_group_df[,c("filepath","group","label")], by="filepath") %>% 
  filter(!is.na(group))
dim(all_imgs_results_gpt2)
table(all_imgs_results_gpt2$group)

# keep only records with group structure
dim(all_imgs_results_big_model)
all_imgs_results_big_model2 = all_imgs_results_big_model %>% 
  left_join(product_group_df[,c("filepath","group")], by = "filepath") %>% 
  filter(!is.na(group))
dim(all_imgs_results_big_model2)
table(all_imgs_results_big_model2$group)

# entract columns for each type of label
col_prd = grep("Product", names(all_imgs_results_big_model2), value = TRUE)
col_weight = grep("Weight", names(all_imgs_results_big_model2), value = TRUE)
col_halal = grep("Halal", names(all_imgs_results_big_model2), value = TRUE)
col_health = grep("Health", names(all_imgs_results_big_model2), value = TRUE)

# group by label and group to obtain average logits for all categories across each lobel
all_imgs_results_big_model3_avglogit = all_imgs_results_big_model2 %>% 
  select(-X, -filepath) %>% 
  group_by(label, group) %>%
  summarise(across(everything(), mean, na.rm = TRUE))

# extract avg model uncertainty using variance
all_imgs_results_big_model3_modelvar = all_imgs_results_big_model2 %>% 
  rowwise() %>%
  mutate(var_halal = var(c_across(c(HalalStatus_Halal, HalalStatus_NonHalal))),
         var_health = var(c_across(c(HealthStatus_Healthy, HealthStatus_NonHealthy))),
         var_product = var(c_across(starts_with("ProductType_"))),
         var_weight = var(c_across(starts_with("Weight_")))) %>%
  ungroup() # Remember to ungroup to avoid unexpected behavior in future operations

all_imgs_results_big_model3_modelvar = all_imgs_results_big_model3_modelvar %>%
  group_by(label, group) %>%
  summarise(
    avg_var_halal = mean(var_halal, na.rm = TRUE),
    avg_var_health = mean(var_health, na.rm = TRUE),
    avg_var_product = mean(var_product, na.rm = TRUE),
    avg_var_weight = mean(var_weight, na.rm = TRUE)
  )

# extract data uncertainty via entropy
# Softmax function to convert logits to probabilities
softmax <- function(logits) {
  exp_logits <- exp(logits)
  probabilities <- exp_logits / sum(exp_logits, na.rm = TRUE)
  return(probabilities)
}

# Function to calculate entropy
calc_entropy <- function(probabilities) {
  # Calculate entropy
  entropy <- -sum(probabilities * log(probabilities), na.rm = TRUE)
  return(entropy)
}

all_imgs_results_big_model3_modelentropy <- all_imgs_results_big_model2 %>%
  rowwise() %>%
  mutate(
    # Convert logits to probabilities for each group and calculate entropy
    entropy_halal = calc_entropy(softmax(c_across(c(HalalStatus_Halal, HalalStatus_NonHalal)))),
    entropy_health = calc_entropy(softmax(c_across(c(HealthStatus_Healthy, HealthStatus_NonHealthy)))),
    entropy_product = calc_entropy(softmax(c_across(starts_with("ProductType_")))),
    entropy_weight = calc_entropy(softmax(c_across(starts_with("Weight_"))))
  ) %>%
  ungroup()

all_imgs_results_big_model3_modelentropy = all_imgs_results_big_model3_modelentropy %>%
  group_by(label, group) %>%
  summarise(
    avg_entropy_halal = mean(entropy_halal, na.rm = TRUE),
    avg_entropy_health = mean(entropy_health, na.rm = TRUE),
    avg_entropy_product = mean(entropy_product, na.rm = TRUE),
    avg_entropy_weight = mean(entropy_weight, na.rm = TRUE)
  )

# aggregate all info
all_imgs_results_big_model3 = all_imgs_results_big_model3_avglogit %>% 
  left_join(all_imgs_results_big_model3_modelvar, by=c("label", "group")) %>% 
  left_join(all_imgs_results_big_model3_modelentropy, by=c("label", "group")) 



###### BAYESIAN MODEL FOR PRODUCT TYPE #######
# process big model file
all_imgs_results_big_model = all_imgs_results_big_model3

# Select columns that start with 'ProductType_'
prdtype_cols <- grep("ProductType_", names(all_imgs_results_big_model), value = TRUE)
prdtype_cols <- sub("ProductType_", "", prdtype_cols)

# Extract true label for product type
all_imgs_results_big_model <- all_imgs_results_big_model %>% 
  mutate(label_prdtype = sub("_.*", "", label))

# Remove 'ProductType_' prefix from column names
names(all_imgs_results_big_model) <- sub("ProductType_", "", names(all_imgs_results_big_model))

# Select label, group, logits and true label
all_imgs_results_big_model = all_imgs_results_big_model[,c("label","group",prdtype_cols,"label_prdtype", "avg_var_product", "avg_entropy_product")]

# process gpt file
all_imgs_results_gpt = all_imgs_results_gpt2
dim(all_imgs_results_gpt)
names(all_imgs_results_gpt)

for(i in 1:length(names(all_imgs_results_gpt))){
  if(i >=3 & i <= 12){
    colnames(all_imgs_results_gpt)[i] = paste0("gpt_", colnames(all_imgs_results_gpt)[i])
  }
}
names(all_imgs_results_gpt)


# romove rows with duplicate filepath
dim(all_imgs_results_gpt)
all_imgs_results_gpt <- all_imgs_results_gpt %>% distinct(filepath, .keep_all = TRUE)
dim(all_imgs_results_gpt)

# check missing values
sum(is.na(all_imgs_results_gpt$gpt_product_type)) > 0
sum(is.na(all_imgs_results_gpt$gpt_image_reflection)) > 0
sum(is.na(all_imgs_results_gpt$gpt_image_clarity)) > 0
sum(is.na(all_imgs_results_gpt$gpt_product_type_confidence)) > 0

# check unique values
unique(all_imgs_results_gpt$gpt_product_type)
all_imgs_results_gpt = all_imgs_results_gpt[!(all_imgs_results_gpt$gpt_product_type %in% c("Unable to determine", "Unable to accurately determine", "Cannot determine from the image", "Unknown")),]
unique(all_imgs_results_gpt$gpt_product_type)

unique(all_imgs_results_gpt$gpt_image_reflection)
unique(all_imgs_results_gpt$gpt_image_clarity)
unique(all_imgs_results_gpt$gpt_product_type_confidence)
dim(all_imgs_results_gpt)

# check if all gpt label-group combinations are inside the big model file
all_imgs_results_gpt$label_group = paste0(all_imgs_results_gpt$label, "-", all_imgs_results_gpt$group)
all_imgs_results_big_model$label_group = paste0(all_imgs_results_big_model$label, "-", all_imgs_results_big_model$group)
sum(all_imgs_results_gpt$label_group %in% all_imgs_results_big_model$label_group) == nrow(all_imgs_results_gpt)
all_imgs_results_gpt$label_group = NULL
all_imgs_results_big_model$label_group = NULL

# left join
total_df = left_join(all_imgs_results_big_model, all_imgs_results_gpt[,colnames(all_imgs_results_gpt)[3:ncol(all_imgs_results_gpt)]], by=c("label", "group"))
dim(total_df)

# remove nonmapped rows
total_df = total_df %>% 
  filter(!is.na(gpt_product_type))
dim(total_df)

# keep only prd types (both row and column wise) based on intersection of predictions
a = unique(total_df$gpt_product_type)
b = unique(total_df$label_prdtype)

total_df$prediction_bigmodel <- prdtype_cols[apply(total_df[,prdtype_cols], 1, which.max)]
c = unique(total_df$prediction_bigmodel)
d = intersect(a, b)
e = intersect(d, c)
unique_lables = e

# select intersected rows and columns
sum(unique_lables %in% prdtype_cols) == length(unique_lables)

total_df = total_df %>% 
  filter(label_prdtype %in% unique_lables,
         gpt_product_type %in% unique_lables,
         prediction_bigmodel %in% unique_lables)
dim(total_df)
total_df = total_df[, c("label", "group", "label_prdtype", unique_lables, 
                        "gpt_product_type", "gpt_image_reflection", "gpt_image_clarity", 
                        "gpt_product_type_confidence", "avg_var_product", "avg_entropy_product")]

# rescale logits into probabilities
softmax <- function(x) {
  e_x <- exp(x - max(x))
  return(e_x / sum(e_x))
}

total_df[,unique_lables] <- t(apply(total_df[,unique_lables], 1, softmax))

##### adhoc - scaling back the probabilities
softmax_with_temp <- function(logits, temperature = 1) {
  exp_logits <- exp(logits / temperature)
  probabilities <- exp_logits / sum(exp_logits)
  return(probabilities)
}

# Apply softmax with higher temperature to reduce gaps
temperature <- 2  # Increasing the temperature makes the output probabilities closer
total_df[,unique_lables] <- t(apply(total_df[,unique_lables], 1, softmax_with_temp, temperature = temperature))

total_df$prediction_bigmodel <- unique_lables[apply(total_df[,unique_lables], 1, which.max)]
dim(total_df)
length(unique(total_df$prediction_bigmodel) %in% unique_lables)
length(unique(total_df$gpt_product_type))
length(unique(total_df$label_prdtype))
length(unique_lables)

sum(total_df$prediction_bigmodel == total_df$label_prdtype) / nrow(total_df)

total_df$X = 1:nrow(total_df)
total_df$true_label = total_df$label_prdtype

sum(total_df$prediction_bigmodel == total_df$true_label) / nrow(total_df)
sum(total_df$gpt_product_type == total_df$true_label) / nrow(total_df)


