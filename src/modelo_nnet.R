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


## Modelo NNET

nnet_param <- expand.grid("size"=1, 
                        "decay" = c(0.8))
                         
# modelo
nnet_mod <- caret::train(x=x_train,
                       y=as.factor(y_train),
                       method="nnet",
                       trControl=train_cv,
                       tuneGrid=nnet_param)

nnet_mod
nnet_mod$results

# matriz de confusion y accuracy
nnet_pred <- predict(nnet_mod, newdata=x_test)
nnet_cm <- caret::confusionMatrix(nnet_pred, as.factor(y_test))
nnet_cm$table
nnet_cm$overall[1]


# Con 5K obs predice bastante bien brand y male pero pesimo Female  Size = 1 Decay = 3
# Con 9K obs tiene 0.546 de accu y predice bastante bien brand      Size = 1 Decay = 3
# Con 12k obs tiene 0.5 y predice muuy bien Female                  Size = 1 Decay = 0.8
# Con 15k obs tiene 0.574 y predice muuy bien Female y Brand        Size = 1 Decay = 0.8
# Con 18k obs tiene 0.55 y predice muuuy bien Female y Brand (mejor)Size = 1 Decay = 0.8

# NNET es util para Brand y Female pero no logra discriminar Male
