library(dplyr)
library(caret)
library(xgboost)
library(tidyverse)
train<- read.csv("C:/Users/vchandramohan/Desktop/Vyshali/House Prediction/train_3.csv")
#test<- read.csv("C:/Users/vchandramohan/Desktop/Vyshali/House Prediction/test_cleaned_1.csv")

train$SalePrice=log1p(train$SalePrice)

outcome <- train$SalePrice
partition <- createDataPartition(y=outcome,
                                 p=.7,
                                 list=F)
training <- train[partition,]
testing <- train[-partition,]
model <- train(
  SalePrice ~., data = training, method = "xgbTree",
  trControl = trainControl("cv", number = 10)
)
varImp(model)
#training1<-training %>% select(SalePrice,livarea_qual,qual_bsmt,nhbd_price_lvl,TotalBath,X2ndFlrSF,AllSF,BsmtFinSF1,year_qual,year_r_qual,LotArea,KitchenQual_mod,X1stFlrSF,GarageArea,OverallGrade,YearRemodAdd,GarageScore,smtUnfSF,OverallCond,SaleType_mod,zone)

#model1<-train( SalePrice ~ .,data=training1,method="xgbTree")

#model=gbm(SalePrice ~ . ,data = training,distribution = "gaussian",n.trees = 10000,
#                 shrinkage = 0.01, interaction.depth = 4)
summary(model)

pred<-predict(model,training)
RMSE(log(training$SalePrice),log(pred))

pred_test<- predict(model,test)
RMSE(log(testing$SalePrice),log(pred_test))

plot(testing$SalePrice/10^4,(testing$SalePrice-pred_test)/10^4)

pred_test_log=expm1(pred_test)

write.csv(pred_test_log,"final_pred_log12.csv")
