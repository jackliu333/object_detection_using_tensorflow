library(dplyr)
library(R2jags)

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Read CSV files
all_imgs_results_gpt <- read.csv("chatgpt_prediction.csv")
all_imgs_results_big_model <- read.csv("all_imgs_results_big_model.csv")

colnames(all_imgs_results_gpt)[2] = "filepath"
colnames(all_imgs_results_gpt)[3:ncol(all_imgs_results_gpt)] = paste0("gpt_", colnames(all_imgs_results_gpt)[3:ncol(all_imgs_results_gpt)])

# align both dataframes rowwise
all_imgs_results_gpt$RowIndexSmall <- seq_len(nrow(all_imgs_results_gpt))
all_imgs_results_big_model$RowIndexBig <- seq_len(nrow(all_imgs_results_big_model))
all_imgs_results_combined = left_join(all_imgs_results_big_model, all_imgs_results_gpt[,colnames(all_imgs_results_gpt)[2:ncol(all_imgs_results_gpt)]], by="filepath")


# Select columns that start with 'ProductType_'
prdtype_cols <- grep("ProductType_", names(all_imgs_results_combined), value = TRUE)

all_imgs_results_combined <- all_imgs_results_combined %>% 
  # select(c('filepath', 'label', 'RowIndexBig', 'RowIndexSmall', 'img_type', all_of(prdtype_cols))) %>% 
  mutate(label_prdtype = sub("_.*", "", label))

# Remove 'ProductType_' prefix from column names
names(all_imgs_results_combined) <- sub("ProductType_", "", names(all_imgs_results_combined))

# Get category names and create a mapping from category names to encoded labels
category_names <- unique(all_imgs_results_combined$label_prdtype)
category_to_encoded <- setNames(seq_along(category_names), category_names)

# big model predictions
all_imgs_results_combined$Prediction_bigmodel <- category_names[apply(all_imgs_results_combined[,category_names], 1, which.max)]
all_imgs_results_combined$Correct_pred_bigmodel = ifelse(all_imgs_results_combined$Prediction == all_imgs_results_combined$label_prdtype, "yes", "no")

# write.csv(all_imgs_results_combined[all_imgs_results_combined$Correct_pred_bigmodel=="no",c("filepath", "Correct_pred_bigmodel")],"big_model_wrong_prediction.csv", row.names = F)

# keep matched imgs
all_imgs_results_combined = all_imgs_results_combined[!is.na(all_imgs_results_combined$gpt_product_type),]

dim(all_imgs_results_combined)

# adhoc analsyis
table(all_imgs_results_combined$label_prdtype)
table(all_imgs_results_combined$label_prdtype[all_imgs_results_combined$Correct_pred_bigmodel=="no"])
table(all_imgs_results_gpt$product_type)

unique_pred_big_model = unique(all_imgs_results_combined$Prediction_bigmodel)
unique_pred_gpt = unique(all_imgs_results_combined$gpt_product_type)
unique_truelabel = unique(all_imgs_results_combined$label_prdtype)

intersect(unique_truelabel, unique_pred_big_model)
intersect(unique_truelabel, unique_pred_gpt)

# overall accuracy
sum(all_imgs_results_combined$Prediction_bigmodel==all_imgs_results_combined$label_prdtype) / nrow(all_imgs_results_combined)
sum(all_imgs_results_combined$gpt_product_type==all_imgs_results_combined$label_prdtype) / nrow(all_imgs_results_combined)

# test accuracy
test_df = all_imgs_results_combined[all_imgs_results_combined$Correct_pred_bigmodel=="no",]
sum(test_df$gpt_product_type==test_df$label_prdtype) / nrow(test_df)

# Label encoding
all_imgs_results_combined$truelabel_encoded <- as.vector(category_to_encoded[as.character(all_imgs_results_combined$label_prdtype)])

# Create missing idx for test set
sampled_data1 <- all_imgs_results_combined %>%
  # filter(truelabel %in% as.vector(category_to_encoded[names(category_to_encoded) %in% c("SweetsChocolatesOthers", "PotatoSticks", "Peanutbutter", "Sugar")])) %>%
  filter(Correct_pred_bigmodel == "yes") %>% 
  group_by(truelabel_encoded) %>%
  sample_n(2)

sampled_data2 <- all_imgs_results_combined %>%
  filter(Correct_pred_bigmodel == "no")
sampled_data2 = sampled_data2[1:15,]

sampled_data = sampled_data1 %>% 
  bind_rows(sampled_data2)

missingidx_bigmodel <- sampled_data$RowIndexBig

# truelabel_missing = all_imgs_results_combined$truelabel[all_imgs_results_combined$RowIndexBig %in% missingidx_bigmodel]
truelabel_missing = c()
for(i in 1:length(missingidx_bigmodel)){
  tmp = all_imgs_results_combined$truelabel_encoded[all_imgs_results_combined$RowIndexBig == missingidx_bigmodel[i]]
  truelabel_missing = c(truelabel_missing, tmp)
}

all_imgs_results_combined$truelabel2 = all_imgs_results_combined$truelabel

all_imgs_results_combined$truelabel[all_imgs_results_combined$RowIndexBig %in% missingidx_bigmodel] = NA

correct_pred_missing_big_model = all_imgs_results_combined$Correct_pred[all_imgs_results_combined$RowIndexBig %in% missingidx_bigmodel]

# testset prediction for big model
pred_missing_big_model = c()
for(i in 1:length(missingidx_bigmodel)){
  tmp = all_imgs_results_combined$Prediction_bigmodel[all_imgs_results_combined$RowIndexBig == missingidx_bigmodel[i]]
  tmp = as.vector(category_to_encoded[as.character(tmp)])
  pred_missing_big_model = c(pred_missing_big_model, tmp)
}
# verify
sum(pred_missing_big_model == truelabel_missing) == sum(correct_pred_missing_big_model == "yes")



# prepare data for Bayesian model
truelabel = all_imgs_results_combined$truelabel
all_imgs_results_combined$row_idx = 1:nrow(all_imgs_results_combined)
# logitscoresA <- as.matrix(sampled_imgs_esults_big_model_prdtype[, category_names])
logitscoresA <- all_imgs_results_combined[, category_names]
ordered_names <- names(category_to_encoded)[order(category_to_encoded)]
logitscoresA = as.matrix(logitscoresA[, ordered_names])

# logitscoresB <- as.matrix(sampled_imgs_results_small_model_prdtype[, category_names])
# logitscoresB = logitscoresB[, ordered_names]
classificationB = all_imgs_results_combined$gpt_product_type
classificationB = as.vector(category_to_encoded[as.character(classificationB)])

trulabel_missing_idx = all_imgs_results_combined$row_idx[is.na(all_imgs_results_combined$truelabel)]

all_imgs_results_combined = all_imgs_results_combined %>% 
  mutate(gpt_prediction_confidence = case_when(gpt_prediction_confidence == "High" ~ 3,
                                               gpt_prediction_confidence == "Medium" ~ 2,
                                               gpt_prediction_confidence == "Low" ~ 1,
                                TRUE ~ 0),
        gpt_image_clarity = case_when(gpt_image_clarity == "High" ~ 3,
                                      gpt_image_clarity == "Medium" ~ 2,
                                      gpt_image_clarity == "Low" ~ 1,
                                      TRUE ~ 0
                            ),
        gpt_image_reflection = case_when(gpt_image_reflection == "High" ~ 3,
                                         gpt_image_reflection == "Medium" ~ 2,
                                         gpt_image_reflection == "Low" ~ 1,
                                TRUE ~ 0))

confidenceB = as.integer(all_imgs_results_combined$gpt_prediction_confidence)
clarityB = as.integer(all_imgs_results_combined$gpt_image_clarity)
reflectionB = as.integer(all_imgs_results_combined$gpt_image_reflection)


# thresholds for confidence level
nConflevels = length(unique(all_imgs_results_combined$gpt_prediction_confidence))
thresh_confidence = rep(NA, nConflevels-1)
thresh_confidence[1] = 1 + 0.5
thresh_confidence[nConflevels-1] = nConflevels-1 + 0.5

# thresholds for confidence level
nClarity = length(unique(all_imgs_results_combined$gpt_image_clarity))
thresh_clarity = rep(NA, nClarity-1)
thresh_clarity[1] = 1 + 0.5
thresh_clarity[nClarity-1] = nClarity-1 + 0.5

# thresholds for reflection level
nReflection = length(unique(all_imgs_results_combined$gpt_image_reflection))
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
truelabel_missing2 = c()
pred_missing_big_model2 = c()
pred_missing_gpt_model2 = c()

for(tmp in trulabel_missing_idx){
  tmp_idx = which(all_imgs_results_combined$row_idx == tmp)
  tmp_true_label = all_imgs_results_combined$truelabel2[tmp_idx]
  tmp_pred_big_label = all_imgs_results_combined$Prediction_bigmodel[tmp_idx]
  tmp_pred_gpt_label = all_imgs_results_combined$gpt_product_type[tmp_idx]
  
  lab = paste0("truelabel[",tmp_idx,"]")
  s = posterior_samples[[1]][,lab]
  
  infer_labels = c(infer_labels, Mode(s))
  truelabel_missing2 = c(truelabel_missing2, tmp_true_label)
  pred_missing_big_model2 = c(pred_missing_big_model2, tmp_pred_big_label)
  pred_missing_gpt_model2 = c(pred_missing_gpt_model2, tmp_pred_gpt_label)
}

infer_labels
truelabel_missing2
pred_missing_big_model
pred_missing_gpt_model

print("Bayes model accuracy:")
sum(infer_labels == truelabel_missing2) / length(truelabel_missing2)
print("Big model accuracy:")
sum(correct_pred_missing_big_model == "yes") / length(truelabel_missing2)
print("Small model accuracy:")
sum(correct_pred_missing_gpt_model == "yes") / length(truelabel_missing2)


# print("Big model accuracy:")
# sum(pred_missing_big_model2 == truelabel_missing2) / length(truelabel_missing2)
# print("Small model accuracy:")
# sum(pred_missing_small_model2 == truelabel_missing2) / length(truelabel_missing2)

# sum(pred_missing_big_model == truelabel_missing) / length(truelabel_missing)
# sum(pred_missing_small_model == truelabel_missing) / length(truelabel_missing)
