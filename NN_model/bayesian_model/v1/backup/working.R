library(dplyr)
library(R2jags)

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Read CSV files
all_imgs_results_small_model <- read.csv("all_imgs_results_small_model.csv")
all_imgs_results_big_model <- read.csv("all_imgs_results_big_model.csv")

# adhoc: remove the last four characters in filepath of small model output
all_imgs_results_small_model$filepath <- substr(all_imgs_results_small_model$filepath, 1, nchar(all_imgs_results_small_model$filepath) - 4)
# adhoc: remove the last four characters in filepath of big model output for new imgs
all_imgs_results_big_model$filepath[all_imgs_results_big_model$img_type=="new"] <- substr(all_imgs_results_big_model$filepath[all_imgs_results_big_model$img_type=="new"], 1, nchar(all_imgs_results_big_model$filepath[all_imgs_results_big_model$img_type=="new"]) - 4)

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
all_imgs_results_big_model_prdtype$Prediction <- category_names[apply(all_imgs_results_big_model_prdtype[,category_names], 1, which.max)]
all_imgs_results_big_model_prdtype$Correct_pred = ifelse(all_imgs_results_big_model_prdtype$Prediction == all_imgs_results_big_model_prdtype$label_prdtype, "yes", "no")

# Sampling ten rows for each category
sampled_imgs_results_big_model_prdtype1 <- all_imgs_results_big_model_prdtype %>%
  # filter(label_prdtype %in% c("NutellaChocolate","PotatoSticks","CornChip","AdultMilk")) %>%
  filter(img_type == "existing") %>% 
  group_by(label_prdtype) %>%
  sample_n(4, replace = FALSE)

sampled_imgs_results_big_model_prdtype2 <- all_imgs_results_big_model_prdtype %>%
  filter(img_type == "new")

sampled_imgs_results_big_model_prdtype = sampled_imgs_results_big_model_prdtype1 %>% 
  bind_rows(sampled_imgs_results_big_model_prdtype2)

# table(sampled_imgs_results_big_model_prdtype$label_prdtype, sampled_imgs_results_big_model_prdtype$Correct_pred)
sampled_imgs_results_small_model_prdtype = all_imgs_results_small_model_prdtype[all_imgs_results_small_model_prdtype$RowIndexSmall %in% sampled_imgs_results_big_model_prdtype$RowIndexSmall,]
# table(sampled_imgs_results_small_model_prdtype$label_prdtype, sampled_imgs_results_small_model_prdtype$Correct_pred)


# Label encoding
sampled_imgs_results_big_model_prdtype$truelabel <- as.vector(category_to_encoded[as.character(sampled_imgs_results_big_model_prdtype$label_prdtype)])

# Create missing idx for test set
# sampled_data <- sampled_imgs_results_big_model_prdtype %>%
#   # filter(truelabel %in% as.vector(category_to_encoded[names(category_to_encoded) %in% c("SweetsChocolatesOthers", "PotatoSticks", "Peanutbutter", "Sugar", "JennyBakery")])) %>%
#   group_by(truelabel) %>%
#   sample_n(2)
sampled_data1 <- sampled_imgs_results_big_model_prdtype %>%
  filter(Correct_pred == "yes") %>% 
  group_by(truelabel) %>%
  sample_n(2)

sampled_data2 <- sampled_imgs_results_big_model_prdtype %>%
  filter(Correct_pred == "no")
sampled_data2 = sampled_data2[1:8,]

sampled_data = sampled_data1 %>% 
  bind_rows(sampled_data2)

missingidx_bigmodel <- sampled_data$RowIndexBig
missingidx_smallmodel <- sampled_data$RowIndexSmall

truelabel_missing = sampled_imgs_results_big_model_prdtype$truelabel[sampled_imgs_results_big_model_prdtype$RowIndexBig %in% missingidx_bigmodel]
trulabel_missing_idx = which(sampled_imgs_results_big_model_prdtype$RowIndexBig %in% missingidx_bigmodel)
sampled_imgs_results_big_model_prdtype$truelabel[sampled_imgs_results_big_model_prdtype$RowIndexBig %in% missingidx_bigmodel] = NA

correct_pred_missing_big_model = sampled_imgs_results_big_model_prdtype$Correct_pred[sampled_imgs_results_big_model_prdtype$RowIndexBig %in% missingidx_bigmodel]
# pred_missing_big_model = sampled_imgs_results_big_model_prdtype$Prediction[sampled_imgs_results_big_model_prdtype$RowIndexBig %in% missingidx_bigmodel]
# pred_missing_big_model = as.vector(category_to_encoded[as.character(pred_missing_big_model)])

correct_pred_missing_small_model = sampled_imgs_results_small_model_prdtype$Correct_pred[sampled_imgs_results_small_model_prdtype$RowIndexSmall %in% missingidx_smallmodel]
# pred_missing_small_model= sampled_imgs_results_small_model_prdtype$Prediction[sampled_imgs_results_small_model_prdtype$RowIndexSmall %in% missingidx_smallmodel]
# pred_missing_small_model = as.vector(category_to_encoded[as.character(pred_missing_small_model)])

# align row level sequence
sampled_imgs_results_small_model_prdtype <- sampled_imgs_results_small_model_prdtype[match(sampled_imgs_results_big_model_prdtype$RowIndexSmall, sampled_imgs_results_small_model_prdtype$RowIndexSmall), ]

# prepare data for Bayesian model
truelabel = sampled_imgs_results_big_model_prdtype$truelabel
logitscoresA <- as.matrix(sampled_imgs_results_big_model_prdtype[, category_names])
logitscoresB <- as.matrix(sampled_imgs_results_small_model_prdtype[, category_names])

N = length(truelabel)
L = dim(logitscoresA)[2]

# Set up the data
model_data <- list(N = N, 
                   L = L, 
                   logitscoresA = logitscoresA, 
                   logitscoresB = logitscoresB, 
                   truelabel = truelabel
)

# Choose the parameters to watch
model_params <- c("muA1", "muA0", "sigmaA", "muB1", "rho")

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
      muB0   <- 0   
      sigmaB <- 1
      rho    ~ dunif(-1,1)
      precmu <- 0.01
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
for(tmp in trulabel_missing_idx){
  lab = paste0("truelabel[",tmp,"]")
  s = posterior_samples[[1]][,lab]
  infer_labels = c(infer_labels, Mode(s))
}

infer_labels
truelabel_missing


print("Bayes model accuracy:")
sum(infer_labels == truelabel_missing) / length(truelabel_missing)
print("Big model accuracy:")
sum(correct_pred_missing_big_model == "yes") / length(truelabel_missing)
print("Small model accuracy:")
sum(correct_pred_missing_small_model == "yes") / length(truelabel_missing)

