library(dplyr)
library(R2jags)
library(rBayesianOptimization)
library(tictoc)
library(foreach)
library(doParallel)

# Read CSV files
all_imgs_results_gpt <- read.csv("GPT_model/chatgpt_prediction_v2 newlabel.csv")
all_imgs_results_big_model <- read.csv("../../NN_model/model_weights/traindatawithin1/all_imgs_results_big_model.csv")
product_group_df = read.csv("../product_group.csv")

######## DATA PROCESSING #######
# add group to gpt predictions 
all_imgs_results_gpt2 = all_imgs_results_gpt %>% 
  rename(filepath = img_filename) %>% 
  left_join(product_group_df[,c("filepath","group","label")], by="filepath") %>% 
  filter(!is.na(group))
# dim(all_imgs_results_gpt2)
# table(all_imgs_results_gpt2$group)

# keep only records with group structure
# dim(all_imgs_results_big_model)
all_imgs_results_big_model2 = all_imgs_results_big_model %>% 
  left_join(product_group_df[,c("filepath","group")], by = "filepath") %>% 
  filter(!is.na(group))
# dim(all_imgs_results_big_model2)
# table(all_imgs_results_big_model2$group)

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

# Select columns that start with 'HalalStatus_'
halal_cols <- grep("HalalStatus_", names(all_imgs_results_big_model), value = TRUE)
halal_cols <- sub("HalalStatus_", "", halal_cols)

# Extract true label for halal
all_imgs_results_big_model$label_halal <- sapply(strsplit(all_imgs_results_big_model$label, "_"), function(x) if(length(x) >= 2) x[3] else NA)

# Remove 'HalalStatus_' prefix from column names
names(all_imgs_results_big_model) <- sub("HalalStatus_", "", names(all_imgs_results_big_model))

# Select label, group, logits and true label
all_imgs_results_big_model = all_imgs_results_big_model[,c("label","group",halal_cols,"label_halal", "avg_var_halal", "avg_entropy_halal")]

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
sum(is.na(all_imgs_results_gpt$gpt_halal)) > 0
sum(is.na(all_imgs_results_gpt$gpt_image_reflection)) > 0
sum(is.na(all_imgs_results_gpt$gpt_image_clarity)) > 0
sum(is.na(all_imgs_results_gpt$gpt_halal_confidence)) > 0

# keep valid halals
unique(all_imgs_results_gpt$gpt_halal)
dim(all_imgs_results_gpt)
all_imgs_results_gpt = all_imgs_results_gpt[all_imgs_results_gpt$gpt_halal %in% halal_cols,]
unique(all_imgs_results_gpt$gpt_halal)
dim(all_imgs_results_gpt)

unique(all_imgs_results_gpt$gpt_image_reflection)
unique(all_imgs_results_gpt$gpt_image_clarity)
unique(all_imgs_results_gpt$gpt_halal_confidence)

# check if all gpt records are inside the big model file
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
  filter(!is.na(gpt_halal))
dim(total_df)

# keep only weights (both row and column wise) based on intersection of predictions
a = unique(total_df$gpt_halal)
b = unique(total_df$label_halal)

total_df$prediction_bigmodel <- halal_cols[apply(total_df[,halal_cols], 1, which.max)]
c = unique(total_df$prediction_bigmodel)
d = intersect(a, b)
e = intersect(d, c)
unique_lables = e
unique_lables

# select intersected rows and columns
sum(unique_lables %in% halal_cols) == length(unique_lables)

total_df = total_df %>% 
  filter(label_halal %in% unique_lables,
         gpt_halal %in% unique_lables,
         prediction_bigmodel %in% unique_lables)
dim(total_df)
total_df = total_df[, c("label", "group", "label_halal", unique_lables, 
                        "gpt_halal", "gpt_image_reflection", "gpt_image_clarity", 
                        "gpt_halal_confidence", "avg_var_halal", "avg_entropy_halal")]


# rescale logits into probabilities
softmax <- function(x) {
  e_x <- exp(x - max(x))
  return(e_x / sum(e_x))
}

total_df[,unique_lables] <- t(apply(total_df[,unique_lables], 1, softmax))

softmax_with_temp <- function(logits, temperature = 1) {
  exp_logits <- exp(logits / temperature)
  probabilities <- exp_logits / sum(exp_logits)
  return(probabilities)
}


Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# function that returns complementarity level of the Bayesian combined model based on temperature and test ratio
calc_bayes_complement <- function(df, temperature, test_ratio, seed){
  set.seed(seed)
    
  # Apply softmax with higher temperature to reduce gaps
  df[,unique_lables] <- t(apply(df[,unique_lables], 1, softmax_with_temp, temperature = temperature))
  
  df$prediction_bigmodel <- unique_lables[apply(df[,unique_lables], 1, which.max)]
  df$X = 1:nrow(df)
  df$true_label = df$label_halal
  
  # category to index mapping
  label_idx_df = data.frame("original_label"=unique_lables,
                            "label_idx"=1:length(unique_lables))
  
  # encode categorical features from gpt
  df = df %>% 
    mutate(gpt_halal_confidence = case_when(gpt_halal_confidence == "High" ~ 3,
                                             gpt_halal_confidence == "Medium" ~ 2,
                                             gpt_halal_confidence == "Low" ~ 1,
                                             TRUE ~ 0),
           gpt_image_clarity = case_when(gpt_image_clarity == "High" ~ 3,
                                         gpt_image_clarity == "Medium" ~ 2,
                                         gpt_image_clarity == "Low" ~ 1,
                                         TRUE ~ 0),
           gpt_image_reflection = case_when(gpt_image_reflection == "High" ~ 3,
                                            gpt_image_reflection == "Medium" ~ 2,
                                            gpt_image_reflection == "Low" ~ 1,
                                            TRUE ~ 0))
  
  # encode categorical predictions and true labels
  df = df %>% 
    left_join(label_idx_df, by=c("true_label"="original_label")) 
  df = df %>% 
    mutate(true_label_encoded = label_idx) %>% 
    select(-label_idx)
  
  df = df %>% 
    left_join(label_idx_df, by=c("prediction_bigmodel"="original_label")) 
  df = df %>% 
    mutate(prediction_bigmodel_encoded = label_idx) %>% 
    select(-label_idx)
  
  df = df %>% 
    left_join(label_idx_df, by=c("gpt_halal"="original_label")) 
  df = df %>% 
    mutate(gpt_product_type_encoded = label_idx) %>% 
    select(-label_idx)
  
  df = df[order(df$gpt_halal_confidence),]
  df$true_label_encoded_raw = df$true_label_encoded
  
  # indicator on gpt and machine correctness
  df$bigmodel_correct = ifelse(df$prediction_bigmodel_encoded==df$true_label_encoded, TRUE, FALSE)
  df$gpt_correct = ifelse(df$gpt_product_type_encoded==df$true_label_encoded, TRUE, FALSE)
  
  # overall accuracy
  print("total accuracy big model:")
  print(sum(df$bigmodel_correct) / nrow(df))
  print("total accuracy gpt:")
  print(sum(df$gpt_correct) / nrow(df))
  print("marginal accuracy big model:")
  print(sum((df$bigmodel_correct) & (!df$gpt_correct)) / nrow(df))
  print("marginal accuracy gpt:")
  print(sum((!df$bigmodel_correct) & (df$gpt_correct)) / nrow(df))
  
  # set test observations to NA 
  df$type = "Train"
  test_idx = sample(df$X, size=as.integer(test_ratio*nrow(df)), replace=F)
  df$type[df$X %in% test_idx] = "Test"
  missingidx = df$X[df$type == "Test"]
  df$true_label_encoded[df$X %in% missingidx] = NA
  sum(is.na(df$true_label_encoded))
  
  # training accuracy 
  # sum(total_df$bigmodel_correct[!is.na(total_df$true_label_encoded)]) / nrow(total_df[!is.na(total_df$true_label_encoded),])
  # sum(total_df$gpt_correct[!is.na(total_df$true_label_encoded)]) / nrow(total_df[!is.na(total_df$true_label_encoded),])
  # sum((total_df$bigmodel_correct[!is.na(total_df$true_label_encoded)]) & (!total_df$gpt_correct[!is.na(total_df$true_label_encoded)])) / nrow(total_df[!is.na(total_df$true_label_encoded),])
  # sum((!total_df$bigmodel_correct[!is.na(total_df$true_label_encoded)]) & (total_df$gpt_correct[!is.na(total_df$true_label_encoded)])) / nrow(total_df[!is.na(total_df$true_label_encoded),])
  
  # prepare data for Bayesian model
  logitscoresA <- df[, unique_lables]
  classificationB = df$gpt_weight_encoded
  truelabel = df$true_label_encoded
  
  confidenceB = as.integer(df$gpt_weight_confidence)
  clarityB = as.integer(df$gpt_image_clarity)
  reflectionB = as.integer(df$gpt_image_reflection)
  
  # thresholds for confidence level
  nConflevels = length(unique(df$gpt_weight_confidence))
  thresh_confidence = rep(NA, nConflevels-1)
  thresh_confidence[1] = 1 + 0.5
  thresh_confidence[nConflevels-1] = nConflevels-1 + 0.5
  
  # thresholds for confidence level
  nClarity = length(unique(df$gpt_image_clarity))
  thresh_clarity = rep(NA, nClarity-1)
  thresh_clarity[1] = 1 + 0.5
  thresh_clarity[nClarity-1] = nClarity-1 + 0.5
  
  # thresholds for reflection level
  nReflection = length(unique(df$gpt_image_reflection))
  thresh_reflection = rep(NA, nReflection-1)
  thresh_reflection[1] = 1 + 0.5
  thresh_reflection[nReflection-1] = nReflection-1 + 0.5
  
  N = length(truelabel)
  L = dim(logitscoresA)[2]
  
  # Set up the data
  model_data <- list(N = N, 
                     L = L, 
                     nConflevels = nConflevels,
                     nClarity = nClarity,
                     nReflection = nReflection,
                     logitscoresA = logitscoresA, 
                     classificationB = classificationB, 
                     confidenceB = confidenceB,
                     clarityB = clarityB,
                     reflectionB = reflectionB,
                     truelabel = truelabel,
                     thresh_confidence = thresh_confidence,
                     thresh_clarity = thresh_clarity,
                     thresh_reflection = thresh_reflection)
  
  # Choose the parameters to watch
  model_params <- c("muA1", "muA0", "sigmaA",
                    "muB1", "rho", "delta_confidence", 
                    "delta_clarity", "delta_reflection")
  
  model_code = "
  model{
    for (i in 1:N) {
      # Prior on truelabel (true label can be observed or latent)
      truelabel[i] ~ dcat( labelprob[1:L] )
      
      # Set the means based on true label
      for (k in 1:L) {
        muA[i,k] <- ifelse( truelabel[i]==k , muA1, muA0 )
        muB[i,k] <- ifelse( truelabel[i]==k , muB1, muB0 )         
      }
      
      # Generate the correlated logit scores for each label k
      for (k in 1:L) {
        # JAGS does not allow partial observations for multivariate normal, so constructing a bivariate normal using normal draws
        logitscoresA[i,k] ~ dnorm( muA[i,k] , 1 / (sigmaA*sigmaA))         
        logitscoresB[i,k] ~ dnorm(  sigmaB * rho * ((logitscoresA[i,k]-muA[i,k])/sigmaA) + muB[i,k] , 1 / (( 1-rho*rho ) * sigmaB * sigmaB ) )                
      }
      
      # Exponentiate
      for (k in 1:L) {
        explogitB[i,k ] = exp( logitscoresB[i,k] ); 
      }
      sumexplogitB[i] = sum( explogitB[i,1:L] )
      
      # And normalize
      for (k in 1:L) {
        probscoresB[i,k ] = explogitB[i,k ] / sumexplogitB[i]
      }
      
      # Compute softmax scores
      for (k in 1:L) {
        softmaxscores[i,k] <- exp( probscoresB[i,k] / tau )
      }
      
      # Generate classification for classifier B
      classificationB[i] ~ dcat( softmaxscores[i,1:L] )	   
      
      # Generate confidence rating for classifier B from an ordered probit model       
      confidenceB[i] ~ dcat(pr_confidenceB[i, 1:nConflevels] )
      pr_confidenceB[i,1] <- pnorm(thresh_confidence[1] , probscoresB[i, classificationB[i]]*delta_confidence , (1/sigmaB^2)*delta_confidence)
      for(k in 2:(nConflevels-1)){
          pr_confidenceB[i,k] <- max(0, pnorm(thresh_confidence[k] , probscoresB[i, classificationB[i]]*delta_confidence , (1/sigmaB^2)*delta_confidence)
                             - pnorm(thresh_confidence[k-1] , probscoresB[i, classificationB[i]]*delta_confidence , (1/sigmaB^2)*delta_confidence))
      }
      pr_confidenceB[i, nConflevels] <- 1 - pnorm(thresh_confidence[nConflevels-1], probscoresB[i, classificationB[i]]*delta_confidence, (1/sigmaB^2)*delta_confidence)
      
      # Generate clarity rating for classifier B from an ordered probit model       
      clarityB[i] ~ dcat(pr_clarityB[i, 1:nClarity] )
      pr_clarityB[i,1] <- pnorm(thresh_clarity[1] , probscoresB[i, classificationB[i]]*delta_clarity, (1/sigmaB^2)*delta_clarity)
      for(k in 2:(nClarity-1)){
          pr_clarityB[i,k] <- max(0, pnorm(thresh_clarity[k] , probscoresB[i, classificationB[i]]*delta_clarity , (1/sigmaB^2)*delta_clarity)
                             - pnorm(thresh_clarity[k-1] , probscoresB[i, classificationB[i]]*delta_clarity , (1/sigmaB^2)*delta_clarity))
      }
      pr_clarityB[i, nClarity] <- 1 - pnorm(thresh_clarity[nClarity-1], probscoresB[i, classificationB[i]]*delta_clarity, (1/sigmaB^2)*delta_clarity)
      
      # Generate reflection rating for classifier B from an ordered probit model       
      reflectionB[i] ~ dcat(pr_reflectionB[i, 1:nReflection] )
      pr_reflectionB[i,1] <- pnorm(thresh_reflection[1] , probscoresB[i, classificationB[i]]*delta_reflection, (1/sigmaB^2)*delta_reflection)
      for(k in 2:(nReflection-1)){
          pr_reflectionB[i,k] <- max(0, pnorm(thresh_reflection[k] , probscoresB[i, classificationB[i]]*delta_reflection , (1/sigmaB^2)*delta_reflection)
                             - pnorm(thresh_reflection[k-1] , probscoresB[i, classificationB[i]]*delta_reflection, (1/sigmaB^2)*delta_reflection))
      }
      pr_reflectionB[i, nReflection] <- 1 - pnorm(thresh_reflection[nReflection-1], probscoresB[i, classificationB[i]]*delta_reflection, (1/sigmaB^2)*delta_reflection)
  
    }
    
    # Uniform prior over labels
    for (k in 1:L) { 
        # Unnormalized
        labelprob[k] <- 1 
    }
    
    # Uniform prior over cutpoints for the ordered probit model
    for ( k in 2:(nConflevels-2) ) {  # 1 and nConflevels-1 are fixed, not stochastic
      thresh_confidence[k] ~ dnorm( k+0.5 , 1/2^2 )
    }
    for ( k in 2:(nConflevels-2) ) {  
      thresh_clarity[k] ~ dnorm( k+0.5 , 1/2^2 )
    }
    for ( k in 2:(nConflevels-2) ) {  
      thresh_reflection[k] ~ dnorm( k+0.5 , 1/2^2 )
    }
    
    # Priors and constants 
    muA1   ~ dnorm(0,precmu)
    muA0   ~ dnorm(0,precmu)
    sigmaA ~ dunif(0.01,bsigma)
    muB1   ~ dnorm(0,precmu)
    muB0   <- 0   
    sigmaB <- 1
    rho    ~ dunif(-1,1)
    tau    <- 0.05
    delta_confidence ~ dunif(1,bdelta)
    delta_clarity ~ dunif(1,bdelta)
    delta_reflection ~ dunif(1,bdelta)
    # delta <- 1
    precmu <- 0.01
    bdelta <- 3
    bsigma <- 0.1
  }
  "
  
  model_run<-jags(data = model_data,
                  parameters.to.save = model_params,
                  model.file = textConnection(model_code),
                  n.iter = 500,
                  n.burnin = 100,
                  n.thin = 1)
  
  posterior_samples <- coda.samples(model_run$model, variable.names=c("truelabel"), n.iter=100)
  
  infer_labels = c()
  true_labels = c()
  gpt_labels = c()
  bigmodel_labels = c()
  
  for(tmp in missingidx){
    tmp_idx = which(df$X == tmp)
    tmp_true_label = df$true_label_encoded_raw[tmp_idx]
    tmp_gpt_label = df$gpt_weight_encoded[tmp_idx]
    tmp_bigmodel_label = df$prediction_bigmodel_encoded[tmp_idx]
    
    lab = paste0("truelabel[",tmp_idx,"]")
    s = posterior_samples[[1]][,lab]
    
    infer_labels = c(infer_labels, Mode(s))
    true_labels = c(true_labels, tmp_true_label)
    gpt_labels = c(gpt_labels, tmp_gpt_label)
    bigmodel_labels = c(bigmodel_labels, tmp_bigmodel_label)
  }
  
  print("total images:")
  print(dim(df)[1])
  print("test images")
  print(nrow(df[df$type=="Test",]))
  print("gpt correct:")
  print(sum(gpt_labels==true_labels) / length(true_labels))
  print("big model correct:")
  print(sum(bigmodel_labels==true_labels) / length(true_labels))
  print("combined correct:")
  print(sum(infer_labels==true_labels) / length(true_labels))
  
  bayes_complementarity = sum(infer_labels==true_labels) / max(sum(gpt_labels==true_labels), sum(bigmodel_labels==true_labels))
    
  return(mean(bayes_complementarity))
}

# calc_bayes_complement(total_df, temperature=0.9, test_ratio=0.7, seed=1)

# Register the parallel backend to use multiple cores
no_cores <- detectCores() - 1  # Use one less than the total number of cores
registerDoParallel(cores=no_cores)

# Prepare a vector of seed values
tic()
seed_values <- c(1, 2, 3)
results <- foreach(seed=seed_values) %dopar% {
  calc_bayes_complement(total_df, temperature=0.9, test_ratio=0.7, seed=seed)
}
stopImplicitCluster()
toc()
mean(sapply(results, function(x) x))



objective_function <- function(temperature, test_ratio) {
  no_cores <- detectCores() - 1  # Use one less than the total number of cores
  registerDoParallel(cores=no_cores)
  
  seed_values <- c(1, 2, 3)
  results <- foreach(seed=seed_values) %dopar% {
    calc_bayes_complement(total_df, temperature=temperature, test_ratio=test_ratio, seed=seed)
  }
  stopImplicitCluster()
  
  list(Score = mean(sapply(results, function(x) x)))
}

bounds <- list(temperature = c(0, 3), test_ratio = c(0.5, 0.9))

set.seed(123) # For reproducibility
bayes_opt_result <- BayesianOptimization(
  FUN = objective_function,
  bounds = bounds,
  init_points = 2,
  n_iter = 5,
  acq = "ei",
  verbose = TRUE
)

bayes_opt_result$Best_Par



              
             
            
                     
      
  
      
          
                         
                    
                              
                                      
                             
                      
                      
                      
            
                                     
                     
     
