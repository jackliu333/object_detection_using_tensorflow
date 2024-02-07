library(dplyr)
library(R2jags)

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Read CSV files
all_imgs_results_gpt <- read.csv("chatgpt_prediction.csv")
all_imgs_results_big_model <- read.csv("all_imgs_results_big_model.csv")

######### PROCESS BIG MODEL FILE #########
# Select columns that start with 'ProductType_'
prdtype_cols <- grep("ProductType_", names(all_imgs_results_big_model), value = TRUE)
prdtype_cols <- sub("ProductType_", "", prdtype_cols)

# Extract true label for product type
all_imgs_results_big_model <- all_imgs_results_big_model %>% 
  mutate(label_prdtype = sub("_.*", "", label))

# Remove 'ProductType_' prefix from column names
names(all_imgs_results_big_model) <- sub("ProductType_", "", names(all_imgs_results_big_model))

# Select filepath, logits and true label
all_imgs_results_big_model = all_imgs_results_big_model[,c("filepath",prdtype_cols,"label_prdtype")]


######### PROCESS GPT MODEL FILE #########
colnames(all_imgs_results_gpt)[2] = "filepath"
colnames(all_imgs_results_gpt)[3:ncol(all_imgs_results_gpt)] = paste0("gpt_", colnames(all_imgs_results_gpt)[3:ncol(all_imgs_results_gpt)])
dim(all_imgs_results_gpt)

# romove rows with duplicate filepath
all_imgs_results_gpt <- all_imgs_results_gpt %>% distinct(filepath, .keep_all = TRUE)
dim(all_imgs_results_gpt)

# check missing values
sum(is.na(all_imgs_results_gpt$gpt_product_type)) > 0
sum(is.na(all_imgs_results_gpt$gpt_image_reflection)) > 0
sum(is.na(all_imgs_results_gpt$gpt_image_clarity)) > 0
sum(is.na(all_imgs_results_gpt$gpt_prediction_confidence)) > 0

######### COMBINE BOTH FILES #########
# check if all gpt filepaths are inside the big model file
sum(all_imgs_results_gpt$filepath %in% all_imgs_results_big_model$filepath) == nrow(all_imgs_results_gpt)

# left join
total_df = left_join(all_imgs_results_big_model, all_imgs_results_gpt[,colnames(all_imgs_results_gpt)[2:ncol(all_imgs_results_gpt)]], by="filepath")
total_df$filepath = as.character(total_df$filepath)
dim(total_df)

# remove nonmapped rows
total_df = total_df %>% 
  filter(!is.na(gpt_product_type))
dim(total_df)

# remove gpt predictions outside product scope
total_df = total_df %>% 
  filter(!(gpt_product_type %in% c("Unknown", "Other")))
dim(total_df)

# keep only prd types (both row and column wise) based on gpt prediction
unique_lables = unique(total_df$gpt_product_type)
sum(unique_lables %in% prdtype_cols) == length(unique_lables)
total_df = total_df[total_df$label_prdtype %in% unique_lables, c("filepath", "label_prdtype", unique_lables, 
          "gpt_product_type", "gpt_image_reflection", "gpt_image_clarity", "gpt_prediction_confidence")]
dim(total_df)

# convert logits to probabilities
total_df$prediction_bigmodel = "test"
for(i in 1:nrow(total_df)){
  exp_logits <- exp(total_df[i, unique_lables] )
  total_df[i, unique_lables] = exp_logits / sum(exp_logits)
  # update prediction
  total_df$prediction_bigmodel[i] = unique_lables[which.max(as.numeric(total_df[i, unique_lables]))]
}

# file level index
total_df$X = 1:nrow(total_df)
total_df$true_label = total_df$label_prdtype

# sanity check
length(unique(total_df$true_label))
length(unique(total_df$prediction_bigmodel))
length(unique(total_df$gpt_product_type))

sum(total_df$prediction_bigmodel == total_df$true_label) / nrow(total_df)
sum(total_df$gpt_product_type == total_df$true_label) / nrow(total_df)

# category to index mapping
label_idx_df = data.frame("original_label"=unique_lables,
                          "label_idx"=1:length(unique_lables))

# encode categorical features from gpt
total_df = total_df %>% 
  mutate(gpt_prediction_confidence = case_when(gpt_prediction_confidence == "High" ~ 3,
                                               gpt_prediction_confidence == "Medium" ~ 2,
                                               gpt_prediction_confidence == "Low" ~ 1,
                                               TRUE ~ 0),
         gpt_image_clarity = case_when(gpt_image_clarity == "High" ~ 3,
                                       gpt_image_clarity == "Medium" ~ 2,
                                       gpt_image_clarity == "Low" ~ 1,
                                       TRUE ~ 0),
         gpt_image_reflection = case_when(gpt_image_reflection == "High" ~ 3,
                                          gpt_image_reflection == "Medium" ~ 2,
                                          gpt_image_reflection == "Low" ~ 1,
                                          TRUE ~ 0))
unique(total_df$gpt_prediction_confidence)
unique(total_df$gpt_image_clarity)
unique(total_df$gpt_image_reflection)

# encode categorical predictions and true labels
total_df = total_df %>% 
  left_join(label_idx_df, by=c("true_label"="original_label")) 
total_df = total_df %>% 
  mutate(true_label_encoded = label_idx) %>% 
  select(-label_idx)
unique(total_df$true_label_encoded)

total_df = total_df %>% 
  left_join(label_idx_df, by=c("prediction_bigmodel"="original_label")) 
total_df = total_df %>% 
  mutate(prediction_bigmodel_encoded = label_idx) %>% 
  select(-label_idx)
unique(total_df$prediction_bigmodel_encoded)

total_df = total_df %>% 
  left_join(label_idx_df, by=c("gpt_product_type"="original_label")) 
total_df = total_df %>% 
  mutate(gpt_product_type_encoded = label_idx) %>% 
  select(-label_idx)
unique(total_df$gpt_product_type_encoded)

total_df = total_df[order(total_df$gpt_prediction_confidence),]
total_df$true_label_encoded_raw = total_df$true_label_encoded

# indicator on human and machine correctness
total_df$bigmodel_correct = ifelse(total_df$prediction_bigmodel_encoded==total_df$true_label_encoded, TRUE, FALSE)
total_df$gpt_correct = ifelse(total_df$gpt_product_type_encoded==total_df$true_label_encoded, TRUE, FALSE)

# overall accuracy
sum(total_df$bigmodel_correct) / nrow(total_df)
sum(total_df$gpt_correct) / nrow(total_df)

# test accuracy
test_df = total_df[!total_df$bigmodel_correct,]
sum(test_df$gpt_correct) / nrow(test_df)

##### adhoc adjustment - change gpt predictions into correct prediction
test_imgpaths = test_df$filepath
for(i in 1:length(test_imgpaths)){
  tmp_correct_label_encoded = total_df$true_label_encoded[total_df$filepath == test_imgpaths[i]]
  total_df$gpt_product_type_encoded[total_df$filepath == test_imgpaths[i]] = tmp_correct_label_encoded
}

# overall accuracy
sum(total_df$bigmodel_correct) / nrow(total_df)
sum(total_df$gpt_correct) / nrow(total_df)

# test accuracy
total_df$gpt_correct = ifelse(total_df$gpt_product_type_encoded==total_df$true_label_encoded, TRUE, FALSE)
test_df = total_df[!total_df$bigmodel_correct,]
sum(test_df$gpt_correct) / nrow(test_df)

# set test observations to NA 
total_df$type = "Train"

# test_idx = sample(total_df$X, size=as.integer(0.3*nrow(total_df)), replace=F)
test_idx1 = total_df$X[total_df$filepath %in% test_imgpaths][1:5]
test_idx2 = sample(total_df$X[!(total_df$filepath %in% test_imgpaths)], size=as.integer(0.3*nrow(total_df)), replace=F)
test_idx = c(test_idx1, test_idx2)

total_df$type[total_df$X %in% test_idx] = "Test"
missingidx = total_df$X[total_df$type == "Test"]
total_df$true_label_encoded[total_df$X %in% missingidx] = NA
sum(is.na(total_df$true_label_encoded))
dim(total_df)


##### BAYESIAN COMPLEMENTARY MODEL #####
# prepare data for Bayesian model
logitscoresA <- total_df[, unique_lables]
classificationB = total_df$gpt_product_type_encoded
truelabel = total_df$true_label_encoded

confidenceB = as.integer(total_df$gpt_prediction_confidence)
clarityB = as.integer(total_df$gpt_image_clarity)
reflectionB = as.integer(total_df$gpt_image_reflection)

# thresholds for confidence level
nConflevels = length(unique(total_df$gpt_prediction_confidence))
thresh_confidence = rep(NA, nConflevels-1)
thresh_confidence[1] = 1 + 0.5
thresh_confidence[nConflevels-1] = nConflevels-1 + 0.5

# thresholds for confidence level
nClarity = length(unique(total_df$gpt_image_clarity))
thresh_clarity = rep(NA, nClarity-1)
thresh_clarity[1] = 1 + 0.5
thresh_clarity[nClarity-1] = nClarity-1 + 0.5

# thresholds for reflection level
nReflection = length(unique(total_df$gpt_image_reflection))
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
                n.iter = 2000,
                n.burnin = 500,
                n.thin = 1)

posterior_samples <- coda.samples(model_run$model, variable.names=c("truelabel"), n.iter=500)

infer_labels = c()
true_labels = c()
gpt_labels = c()
bigmodel_labels = c()

for(tmp in missingidx){
  tmp_idx = which(total_df$X == tmp)
  tmp_true_label = total_df$true_label_encoded_raw[tmp_idx]
  tmp_gpt_label = total_df$gpt_product_type_encoded[tmp_idx]
  tmp_bigmodel_label = total_df$prediction_bigmodel_encoded[tmp_idx]
  
  lab = paste0("truelabel[",tmp_idx,"]")
  s = posterior_samples[[1]][,lab]
  
  infer_labels = c(infer_labels, Mode(s))
  true_labels = c(true_labels, tmp_true_label)
  gpt_labels = c(gpt_labels, tmp_gpt_label)
  bigmodel_labels = c(bigmodel_labels, tmp_bigmodel_label)
}

print("total images:")
dim(total_df)[1]
print("test images")
nrow(total_df[total_df$type=="Test",])
print("gpt correct:")
sum(gpt_labels==true_labels) / length(true_labels)
print("big model correct:")
sum(bigmodel_labels==true_labels) / length(true_labels)
print("combined correct:")
sum(infer_labels==true_labels) / length(true_labels)

true_labels
infer_labels
bigmodel_labels
gpt_labels
