# KNN
# corre modelos

# librerias y funciones ---------------------------------------------------
source("src/funciones.R")
source("src/load_librerias.R")


# bases  ------------------------------------------------------------------

### TRAIN
base_tv <- readRDS(file="data/final/base_train_validation.rds")
# predictores train
x_train <- base_tv %>% dplyr::select(-gender) %>% dplyr::mutate_if(is.character, is.factor)
# variable respuesta train
y_train <- base_tv$gender

### TEST
base_test <- readRDS(file="data/final/base_test.rds")
# predictores test
x_test <- base_test %>% dplyr::select(-gender)  %>% dplyr::mutate_if(is.character, is.factor)
# variable respuesta test
y_test <- base_test$gender

#  train-test para probar modelos solo con variables words
text_train <- x_train %>% dplyr::select(dplyr::starts_with("t_"),
                                        dplyr::starts_with("d_"))
text_test <- x_test %>% dplyr::select(dplyr::starts_with("t_"),
                                      dplyr::starts_with("d_"))

x_train$color_link %>% table(useNA="always")



library(caret)

# VALIDATION PARAMETERS

# k-fold CV
train_cv <- caret::trainControl(method="cv",
                                number=5,
                                classProbs=T)
# parametros definidos por usuario sin validation:
train_simple <- caret::trainControl(method="none", 
                                    classProbs=T)

## Base numerica

x_train_knn <- x_train %>% dplyr::select(-c(user_timezone,color_link, color_side))


## PARAMETROS PARA KKNN
knn_param <- expand.grid("kmax"=9, 
                         "distance"=1,
                         "kernel"=c("triangular","rank")) 

# modelo KKNN
kknn_mod <- caret::train(x=x_train_knn[1:1000,],
                        y=as.factor(y_train[1:1000]),
                        method="kknn",
                        trControl=train_cv,
                        tuneGrid = knn_param)
# Esta es la mejor configuracion que encontramos para KKNN. Aprox 0.45 en train y 0.38 en test.


kknn_mod

pred_kknn <- predict(kknn_mod, newdata = x_test)
conf_kknn <-  caret::confusionMatrix(pred_kknn, as.factor(y_test))
conf_kknn$table
conf_kknn$overall

# Modelo KNN

# Este es con KNN (una K menos)
knn_mod <- caret::train(x=x_train_knn[1:1000,],
                        y=as.factor(y_train[1:1000]),
                        method="knn",
                        trControl=train_cv)

knn_mod
pred_knn <- predict(knn_mod, newdata = x_test)
conf_knn <-  caret::confusionMatrix(pred_knn, as.factor(y_test))
conf_knn$table
conf_knn$overall

# Menos accuracy en train 0.4 pero mayor en test 0.42