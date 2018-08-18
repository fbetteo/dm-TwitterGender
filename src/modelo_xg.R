## Neural Net

# corre modelos

# librerias y funciones ---------------------------------------------------
source("src/funciones.R")
source("src/load_librerias.R")


# bases  ------------------------------------------------------------------

### TRAIN
base_tv <- readRDS(file="data/final/base_train_validation.rds")

# predictores train
x_train <- base_tv %>% dplyr::select(-gender) %>% dplyr::mutate_if(is.character, as.factor) 
# variable respuesta train
y_train <- base_tv$gender

### TEST
base_test <- readRDS(file="data/final/base_test.rds")
# predictores test
x_test <- base_test %>% dplyr::select(-gender)  %>% dplyr::mutate_if(is.character, as.factor)
# variable respuesta test
y_test <- base_test$gender

#  train-test para probar modelos solo con variables words
text_train <- x_train %>% dplyr::select(dplyr::starts_with("t_"),
                                        dplyr::starts_with("d_"))
text_test <- x_test %>% dplyr::select(dplyr::starts_with("t_"),
                                      dplyr::starts_with("d_"))

###############

# validation methods ------------------------------------------------------
# repeated CV y grid search (ver bien qué es)
train_rcv <- caret::trainControl(method = "repeatedcv",
                                 number = 10,
                                 repeats = 3,
                                 search = "grid",
                                 # este lo meto para poder calcular ROC (ver doc caret)
                                 classProbs=T)
# k-fold CV
train_cv <- caret::trainControl(method="cv",
                                number=5,
                                classProbs=T)
# parametros definidos por usuario sin validation:
train_simple <- caret::trainControl(method="none", 
                                    classProbs=T)


## Base Numerica
x_train_xg <- x_train %>% dplyr::select(-c(user_timezone,color_link, color_side))
x_test_xg <- x_test %>% dplyr::select(-c(user_timezone,color_link, color_side))
## Modelo XGBOOST

xg_param <- expand.grid("nrounds" = c(10,15), 
                          "lambda" = c(0,1,2,3),
                        "alpha" = c(0,1),
                        "eta" = c(0.01))

# modelo
xg_mod <- caret::train(x=x_train_xg,
                         y=as.factor(y_train),
                         method="xgbLinear",
                         trControl=train_cv,
                         tuneGrid=xg_param)

xg_mod
xg_mod$results

# matriz de confusion y accuracy
xg_pred <- predict(xg_mod, newdata=x_test_xg)
xg_cm <- caret::confusionMatrix(xg_pred, as.factor(y_test))
xg_cm$table
xg_cm$overall[1]


# A mas observaciones, mejor accuracy en Test
# Llega a 0.61 en test,bastante equilibrado en lo que estima


# cOn nrounds = 15, lambda = 0, alpha = 1, eta = 0.01 llega a 0.63 
# mejor Brand y Females que Male como todos...