library(dplyr)
library(readr)

# df = read_csv("../video_recognition/master_list.csv")
df = read_csv("../NN_model/master_list.csv")
all_labels = unique(df$label)
all_prdtypes = unique(df$ProductType)
all_weights = unique(df$Weight)
all_halals = unique(df$HalalStatus)
all_healths = unique(df$HealthStatus)


