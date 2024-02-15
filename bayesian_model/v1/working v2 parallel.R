library(dplyr)
library(R2jags)

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Read CSV files
all_imgs_results_small_model <- read.csv("all_imgs_results_small_model.csv")
all_imgs_results_big_model <- read.csv("all_imgs_results_big_model.csv")

# # adhoc: remove the last four characters in filepath of small model output
# all_imgs_results_small_model$filepath <- substr(all_imgs_results_small_model$filepath, 1, nchar(all_imgs_results_small_model$filepath) - 4)
# # adhoc: remove the last four characters in filepath of big model output for new imgs
# all_imgs_results_big_model$filepath[all_imgs_results_big_model$img_type=="new"] <- substr(all_imgs_results_big_model$filepath[all_imgs_results_big_model$img_type=="new"], 1, nchar(all_imgs_results_big_model$filepath[all_imgs_results_big_model$img_type=="new"]) - 4)

# align both dataframes rowwise
all_imgs_results_small_model$RowIndexSmall <- seq_len(nrow(all_imgs_results_small_model))
all_imgs_results_big_model$RowIndexBig <- seq_len(nrow(all_imgs_results_big_model))
all_imgs_results_big_model = left_join(all_imgs_results_big_model, all_imgs_results_small_model[,c("filepath", "RowIndexSmall")], by="filepath")

# Select columns that start with 'ProductType_'
prdtype_cols <- grep("ProductType_", names(all_imgs_results_big_model), value = TRUE)

# Subset data frames
all_imgs_results_small_model_prdtype <- all_imgs_results_small_model %>% 
  select(c('filepath', 'label', 'RowIndexSmall', 'img_type', all_of(prdtype_cols))) %>% 
  mutate(label_prdtype = sub("_.*", "", label))

all_imgs_results_big_model_prdtype <- all_imgs_results_big_model %>% 
  select(c('filepath', 'label', 'RowIndexBig', 'RowIndexSmall', 'img_type', all_of(prdtype_cols))) %>% 
  mutate(label_prdtype = sub("_.*", "", label))

# Remove 'ProductType_' prefix from column names
names(all_imgs_results_small_model_prdtype) <- sub("ProductType_", "", names(all_imgs_results_small_model_prdtype))
names(all_imgs_results_big_model_prdtype) <- sub("ProductType_", "", names(all_imgs_results_big_model_prdtype))

# Get category names and create a mapping from category names to encoded labels
category_names <- unique(all_imgs_results_big_model_prdtype$label_prdtype)
category_to_encoded <- setNames(seq_along(category_names), category_names)


# adhoc
# all_imgs_results_small_model_prdtype = all_imgs_results_small_model_prdtype[!(all_imgs_results_small_model_prdtype$label_prdtype %in% c("MiloPowder","ChilliSauce")),]
# all_imgs_results_big_model_prdtype = all_imgs_results_big_model_prdtype[!(all_imgs_results_big_model_prdtype$label_prdtype %in% c("MiloPowder","ChilliSauce")),]

all_imgs_results_small_model_prdtype$Prediction <- category_names[apply(all_imgs_results_small_model_prdtype[,category_names], 1, which.max)]
all_imgs_results_small_model_prdtype$Correct_pred = ifelse(all_imgs_results_small_model_prdtype$Prediction == all_imgs_results_small_model_prdtype$label_prdtype, "yes", "no")
print("Small model prediction accuracy:")
print(sum(all_imgs_results_small_model_prdtype$Correct_pred == "yes") / nrow(all_imgs_results_small_model_prdtype))

all_imgs_results_big_model_prdtype$Prediction <- category_names[apply(all_imgs_results_big_model_prdtype[,category_names], 1, which.max)]
all_imgs_results_big_model_prdtype$Correct_pred = ifelse(all_imgs_results_big_model_prdtype$Prediction == all_imgs_results_big_model_prdtype$label_prdtype, "yes", "no")
print("Big model prediction accuracy:")
print(sum(all_imgs_results_big_model_prdtype$Correct_pred == "yes") / nrow(all_imgs_results_big_model_prdtype))



# Sampling a few rows for each category for the big model
sampled_imgs_results_big_model_prdtype1 <- all_imgs_results_big_model_prdtype %>%
  # filter(label_prdtype %in% c("NutellaChocolate","PotatoSticks","CornChip","AdultMilk")) %>%
  filter(img_type == "existing") %>% 
  group_by(label_prdtype) %>%
  sample_n(4, replace = FALSE)

sampled_imgs_results_big_model_prdtype2 <- all_imgs_results_big_model_prdtype %>%
  filter(img_type == "new")

sampled_imgs_results_big_model_prdtype = sampled_imgs_results_big_model_prdtype1 %>% 
  bind_rows(sampled_imgs_results_big_model_prdtype2)
table(sampled_imgs_results_big_model_prdtype$Correct_pred)

# Sampling a few rows for each category for the big model
sampled_imgs_results_small_model_prdtype = all_imgs_results_small_model_prdtype[all_imgs_results_small_model_prdtype$RowIndexSmall %in% sampled_imgs_results_big_model_prdtype$RowIndexSmall,]
table(sampled_imgs_results_small_model_prdtype$Correct_pred)

# Label encoding
sampled_imgs_results_big_model_prdtype$truelabel <- as.vector(category_to_encoded[as.character(sampled_imgs_results_big_model_prdtype$label_prdtype)])

# Create missing idx for test set
# sampled_data <- sampled_imgs_results_big_model_prdtype %>%
#   # filter(truelabel %in% as.vector(category_to_encoded[names(category_to_encoded) %in% c("SweetsChocolatesOthers", "PotatoSticks", "Peanutbutter", "Sugar", "JennyBakery")])) %>%
#   group_by(truelabel) %>%
#   sample_n(2)
sampled_data1 <- sampled_imgs_results_big_model_prdtype %>%
  filter(truelabel %in% as.vector(category_to_encoded[names(category_to_encoded) %in% c("SweetsChocolatesOthers", "PotatoSticks", "Peanutbutter", "Sugar")])) %>%
  filter(Correct_pred == "yes") %>% 
  group_by(truelabel) %>%
  sample_n(2)

sampled_data2 <- sampled_imgs_results_big_model_prdtype %>%
  filter(Correct_pred == "no")
sampled_data2 = sampled_data2[1:6,]

sampled_data = sampled_data1 %>% 
  bind_rows(sampled_data2)

missingidx_bigmodel <- sampled_data$RowIndexBig
missingidx_smallmodel <- sampled_data$RowIndexSmall

# truelabel_missing = sampled_imgs_results_big_model_prdtype$truelabel[sampled_imgs_results_big_model_prdtype$RowIndexBig %in% missingidx_bigmodel]
truelabel_missing = c()
for(i in 1:length(missingidx_bigmodel)){
  tmp = sampled_imgs_results_big_model_prdtype$truelabel[sampled_imgs_results_big_model_prdtype$RowIndexBig == missingidx_bigmodel[i]]
  truelabel_missing = c(truelabel_missing, tmp)
}

sampled_imgs_results_big_model_prdtype$truelabel2 = sampled_imgs_results_big_model_prdtype$truelabel

sampled_imgs_results_big_model_prdtype$truelabel[sampled_imgs_results_big_model_prdtype$RowIndexBig %in% missingidx_bigmodel] = NA

correct_pred_missing_big_model = sampled_imgs_results_big_model_prdtype$Correct_pred[sampled_imgs_results_big_model_prdtype$RowIndexBig %in% missingidx_bigmodel]

# testset prediction for big model
pred_missing_big_model = c()
for(i in 1:length(missingidx_bigmodel)){
  tmp = sampled_imgs_results_big_model_prdtype$Prediction[sampled_imgs_results_big_model_prdtype$RowIndexBig == missingidx_bigmodel[i]]
  tmp = as.vector(category_to_encoded[as.character(tmp)])
  pred_missing_big_model = c(pred_missing_big_model, tmp)
}
# verify
sum(pred_missing_big_model == truelabel_missing) == sum(correct_pred_missing_big_model == "yes")

correct_pred_missing_small_model = sampled_imgs_results_small_model_prdtype$Correct_pred[sampled_imgs_results_small_model_prdtype$RowIndexSmall %in% missingidx_smallmodel] 

pred_missing_small_model = c()
for(i in 1:length(missingidx_smallmodel)){
  tmp = sampled_imgs_results_small_model_prdtype$Prediction[sampled_imgs_results_small_model_prdtype$RowIndexSmall == missingidx_smallmodel[i]]
  tmp = as.vector(category_to_encoded[as.character(tmp)])
  pred_missing_small_model = c(pred_missing_small_model, tmp)
}
# verify
sum(pred_missing_small_model == truelabel_missing) == sum(correct_pred_missing_small_model == "yes")

# # align row level sequence
# sampled_imgs_results_small_model_prdtype <- sampled_imgs_results_small_model_prdtype[match(sampled_imgs_results_big_model_prdtype$RowIndexSmall, sampled_imgs_results_small_model_prdtype$RowIndexSmall), ]

sampled_imgs_results_small_model_prdtype$Prediction_smallmodel = sampled_imgs_results_small_model_prdtype$Prediction
sampled_imgs_results_big_model_prdtype = sampled_imgs_results_big_model_prdtype %>% 
  left_join(sampled_imgs_results_small_model_prdtype[,c("RowIndexSmall","Prediction_smallmodel")], by="RowIndexSmall") 


# prepare data for Bayesian model
truelabel = sampled_imgs_results_big_model_prdtype$truelabel
sampled_imgs_results_big_model_prdtype$row_idx = 1:nrow(sampled_imgs_results_big_model_prdtype)
# logitscoresA <- as.matrix(sampled_imgs_esults_big_model_prdtype[, category_names])
logitscoresA <- sampled_imgs_results_big_model_prdtype[, category_names]
ordered_names <- names(category_to_encoded)[order(category_to_encoded)]
logitscoresA = as.matrix(logitscoresA[, ordered_names])

logitscoresB <- sampled_imgs_results_small_model_prdtype[, category_names]
logitscoresB = as.matrix(logitscoresB[, ordered_names])
# classificationB = sampled_imgs_results_big_model_prdtype$Prediction_smallmodel
# classificationB = as.vector(category_to_encoded[as.character(classificationB)])

# trulabel_missing_idx = which(is.na(sampled_imgs_results_big_model_prdtype$truelabel))
trulabel_missing_idx = sampled_imgs_results_big_model_prdtype$row_idx[is.na(sampled_imgs_results_big_model_prdtype$truelabel)]

# convert prediction to index
sampled_imgs_results_big_model_prdtype$Prediction_bigmodel_idx = as.vector(category_to_encoded[as.character(sampled_imgs_results_big_model_prdtype$Prediction)])
sampled_imgs_results_big_model_prdtype$Prediction_smallmodel_idx = as.vector(category_to_encoded[as.character(sampled_imgs_results_big_model_prdtype$Prediction_smallmodel)])

N = length(truelabel)
L = dim(logitscoresA)[2]

# Set up the data
model_data <- list(N = N, 
                   L = L, 
                   logitscoresA = logitscoresA, 
                   logitscoresB = logitscoresB,
                   # classificationB = classificationB, 
                   truelabel = truelabel
)

# Choose the parameters to watch
model_params <- c("muA1", "muA0", "sigmaA", "muB1", "muB0", "rho")

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

      }
      
      # Uniform prior over labels
      for (k in 1:L) { 
          # Unnormalized
          labelprob[k] <- 1 
      }
      
      # Priors and constants 
      muA1   ~ dnorm(0,precmu)
      muA0   ~ dnorm(0,precmu)
      sigmaA ~ dunif(0.01,bsigma)
      muB1   ~ dnorm(0,precmu)
      muB0   ~ dnorm(0,precmu)
      sigmaB <- 1
      rho    ~ dunif(-1,1)
      tau    <- 0.05
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
pred_missing_small_model2 = c()

for(tmp in trulabel_missing_idx){
  tmp_idx = which(sampled_imgs_results_big_model_prdtype$row_idx == tmp)
  tmp_true_label = sampled_imgs_results_big_model_prdtype$truelabel2[tmp_idx]
  tmp_pred_big_label = sampled_imgs_results_big_model_prdtype$Prediction_bigmodel_idx[tmp_idx]
  tmp_pred_small_label = sampled_imgs_results_big_model_prdtype$Prediction_smallmodel_idx[tmp_idx]
  
  lab = paste0("truelabel[",tmp_idx,"]")
  s = posterior_samples[[1]][,lab]
  
  infer_labels = c(infer_labels, Mode(s))
  truelabel_missing2 = c(truelabel_missing2, tmp_true_label)
  pred_missing_big_model2 = c(pred_missing_big_model2, tmp_pred_big_label)
  pred_missing_small_model2 = c(pred_missing_small_model2, tmp_pred_small_label)
}

infer_labels
truelabel_missing2
pred_missing_big_model
pred_missing_small_model

print("Bayes model accuracy:")
sum(infer_labels == truelabel_missing2) / length(truelabel_missing2)
print("Big model accuracy:")
sum(correct_pred_missing_big_model == "yes") / length(truelabel_missing2)
print("Small model accuracy:")
sum(correct_pred_missing_small_model == "yes") / length(truelabel_missing2)


# print("Big model accuracy:")
# sum(pred_missing_big_model2 == truelabel_missing2) / length(truelabel_missing2)
# print("Small model accuracy:")
# sum(pred_missing_small_model2 == truelabel_missing2) / length(truelabel_missing2)

# sum(pred_missing_big_model == truelabel_missing) / length(truelabel_missing)
# sum(pred_missing_small_model == truelabel_missing) / length(truelabel_missing)

