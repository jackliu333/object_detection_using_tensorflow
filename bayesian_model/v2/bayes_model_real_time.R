library(jsonlite)
library(dplyr)
library(R2jags)

######## DATA LOADING #######
args <- commandArgs(trailingOnly = TRUE)

# Parse JSON
big_model_pred <- fromJSON(args[1])
gpt_pred <- fromJSON(args[2])

# Coerce to DataFrame if necessary
big_model_pred <- as.data.frame(big_model_pred)
gpt_pred <- as.data.frame(gpt_pred)

# cat("Structure after coercion to DataFrame:\n")
# print(str(gpt_pred))

######## DATA PROCESSING #######
# entract columns for each type of label
col_prd = grep("Product", names(big_model_pred), value = TRUE)
# print(col_prd)

big_model_pred["dummy_var"] = 1

# group by label and group to obtain average logits for all categories across each lobel
big_model_pred = big_model_pred %>% 
  select(dummy_var, all_of(col_prd)) %>% 
  group_by(dummy_var) %>%
  summarise(across(everything(), mean, na.rm = TRUE))
# str(big_model_pred)


###### BAYESIAN MODEL FOR PRODUCT TYPE #######
# Select columns that start with 'ProductType_'
prdtype_cols <- grep("ProductType_", names(big_model_pred), value = TRUE)
prdtype_cols <- sub("ProductType_", "", prdtype_cols)
# prdtype_cols

# Remove 'ProductType_' prefix from column names
names(big_model_pred) <- sub("ProductType_", "", names(big_model_pred))

# Select label, group, logits and true label
big_model_pred = big_model_pred[,prdtype_cols]

# process gpt file
for(i in 1:length(names(gpt_pred))){
  colnames(gpt_pred)[i] = paste0("gpt_", colnames(gpt_pred)[i])
}
# names(gpt_pred)


# check unique values
# unique(gpt_pred$gpt_product_type)
gpt_pred = gpt_pred[!(gpt_pred$gpt_product_type %in% c("Unable to determine", "Unable to accurately determine", "Cannot determine from the image", "Unknown")),]
# unique(gpt_pred$gpt_product_type)

# left join
total_df = cbind(big_model_pred, gpt_pred)
# dim(total_df)

# rescale logits into probabilities
softmax <- function(x) {
  e_x <- exp(x - max(x))
  return(e_x / sum(e_x))
}

unique_lables = prdtype_cols
# unique_lables
total_df[,unique_lables] <- t(apply(total_df[,unique_lables], 1, softmax))
# total_df

softmax_with_temp <- function(logits, temperature = 1) {
  exp_logits <- exp(logits / temperature)
  probabilities <- exp_logits / sum(exp_logits)
  return(probabilities)
}

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Choose the parameters to watch
model_params <- c("muA1", "muA0", "sigmaA",
                  "muB1", "rho", "delta_confidence", 
                  "delta_clarity", "delta_reflection")

nConflevels = 3
nClarity = 3
nReflection = 3

# thresholds for confidence level
thresh_confidence = rep(NA, nConflevels-1)
thresh_confidence[1] = 1 + 0.5
thresh_confidence[nConflevels-1] = nConflevels-1 + 0.5

# thresholds for confidence level
thresh_clarity = rep(NA, nClarity-1)
thresh_clarity[1] = 1 + 0.5
thresh_clarity[nClarity-1] = nClarity-1 + 0.5

# thresholds for reflection level
thresh_reflection = rep(NA, nReflection-1)
thresh_reflection[1] = 1 + 0.5
thresh_reflection[nReflection-1] = nReflection-1 + 0.5
# dim(total_df)
total_df$prediction_bigmodel <- unique_lables[apply(total_df[,unique_lables], 1, which.max)]
# total_df

# category to index mapping
label_idx_df = read.csv("label_idx_df.csv", stringsAsFactors = F)

# encode categorical features from gpt
total_df = total_df %>% 
  mutate(gpt_product_type_confidence = case_when(gpt_product_type_confidence == "High" ~ 3,
                                                 gpt_product_type_confidence == "Medium" ~ 2,
                                                 gpt_product_type_confidence == "Low" ~ 1,
                                                 TRUE ~ 0),
         gpt_image_clarity = case_when(gpt_image_clarity == "High" ~ 3,
                                       gpt_image_clarity == "Medium" ~ 2,
                                       gpt_image_clarity == "Low" ~ 1,
                                       TRUE ~ 0),
         gpt_image_reflection = case_when(gpt_image_reflection == "High" ~ 3,
                                          gpt_image_reflection == "Medium" ~ 2,
                                          gpt_image_reflection == "Low" ~ 1,
                                          TRUE ~ 0))

# total_df

# encode categorical predictions and true labels
total_df = total_df %>% 
  left_join(label_idx_df, by=c("prediction_bigmodel"="original_label")) 
total_df = total_df %>% 
  mutate(prediction_bigmodel_encoded = label_idx) %>% 
  select(-label_idx)

total_df = total_df %>% 
  left_join(label_idx_df, by=c("gpt_product_type"="original_label")) 
total_df = total_df %>% 
  mutate(gpt_product_type_encoded = label_idx) %>% 
  select(-label_idx)

total_df = total_df[order(total_df$gpt_product_type_confidence),]
# total_df

seed = 1
set.seed(seed)
    
# Apply softmax with higher temperature to reduce gaps
temperature = 0.9
total_df[,unique_lables] <- t(apply(total_df[,unique_lables], 1, softmax_with_temp, temperature = temperature))
# total_df

# Updated Bayesian model
model_code_new <- "model{
  delta_clarity ~ dnorm(1.006057, 23086.457436)
  delta_confidence ~ dnorm(1.005532, 28996.730910)
  delta_reflection ~ dnorm(1.076348, 232.270488)
  muA0 ~ dnorm(0.027213, 91566129.707946)
  muA1 ~ dnorm(0.074602, 2358440.127863)
  muB1 ~ dnorm(4.278002, 97.927607)
  rho ~ dnorm(-0.086236, 459.168275)
  sigmaA ~ dunif(0.009988, 0.010014)


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
  
  # add constants
  muB0   <- 0   
  sigmaB <- 1
  tau    <- 0.05
  precmu <- 0.01
  bdelta <- 3
  bsigma <- 0.1
}"

# Set up the data
i = 1
test_logitscoresA = total_df[i, unique_lables]
test_classificationB = total_df$gpt_product_type_encoded[i]
test_confidenceB = total_df$gpt_product_type_confidence[i]
test_clarityB = total_df$gpt_image_clarity[i]
test_reflectionB = total_df$gpt_image_reflection[i]
# test_reflectionB

model_data_new <- list(N = 1, 
                       L = length(unique_lables), 
                       nConflevels = nConflevels,
                       nClarity = nClarity,
                       nReflection = nReflection,
                       logitscoresA = test_logitscoresA, 
                       classificationB = test_classificationB, 
                       confidenceB = test_confidenceB,
                       clarityB = test_clarityB,
                       reflectionB = test_reflectionB,
                       truelabel = NA,
                       thresh_confidence = thresh_confidence,
                       thresh_clarity = thresh_clarity,
                       thresh_reflection = thresh_reflection)

# prediction for new data
model_run_new <- jags(data = model_data_new,
                      model.file = textConnection(model_code_new),  
                      n.chains = 3,
                      n.iter = 500,
                      n.burnin = 100,
                      n.thin = 1,
                      parameters.to.save = model_params,
                      quiet=TRUE)


# Extract posterior samples for inference
posterior_samples_new <- coda.samples(model_run_new$model, variable.names = c("truelabel"), n.iter = 100)

bayes_pred = Mode(posterior_samples_new[[1]])
bayes_pred_label = label_idx_df[label_idx_df$label_idx==bayes_pred,"original_label"]
# bayes_pred_label

######## REPORTING #######
# Now df should be a proper DataFrame
# Get the shape of the DataFrame
# shape <- dim(df)

# Return the shape as JSON
cat(toJSON(list(pred = bayes_pred_label)))
