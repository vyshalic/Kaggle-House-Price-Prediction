library(data.table)
library(FeatureHashing)
library(Matrix)
library(xgboost)
require(randomForest)
require(caret)
require(dplyr)
require(ggplot2)
library(pROC)
library(stringr)
library(dummies)
library(Metrics)
library(kernlab)
library(mlbench)

#names(train)
train<-read.csv("C:/Users/vchandramohan/Desktop/Vyshali/House Prediction/train.csv",stringsAsFactors = F)


train$zone[train$MSZoning %in% c("FV")] <- 4
train$zone[train$MSZoning %in% c("RL")] <- 3
train$zone[train$MSZoning %in% c("RH","RM")] <- 2
train$zone[train$MSZoning %in% c("C (all)")] <- 1

# table(train$Street)
train$StreetPaved[train$Street=='Pave']<-1
train$StreetPaved[train$Street!='Pave']<-0

# table(train$Alley)
train$AlleyPaved[train$Alley=='Pave']<-1
train$AlleyPaved[train$Alley!='Pave']<-0
train$AlleyPaved[is.na(train$Alley)]<-0

# table(train$LotShape)
train$LotReg[train$LotShape=='Reg']<-1
train$LotReg[train$LotShape!='Reg']<-0

# table(train$LandContour)
train$LandContourLvl[train$LandContour=='Lvl']<-1
train$LandContourLvl[train$LandContour!='Lvl']<-0

# table(train$Utilities)
train$AllUtilities[train$Utilities=='AllPub']<-1
train$AllUtilities[train$Utilities!='AllPub']<-0

# table(train$LotConfig)
train$culdesac_fr3[train$LotConfig %in% c('CulDSac','FR3')]<-1 #CulDeSac and FR3 have highest avg sale price
train$culdesac_fr3[!train$LotConfig %in% c('CulDSac','FR3')]<-0

# table(train$LandSlope)
train$GtlLandSlope[train$LandSlope=='Gtl']<-1
train$GtlLandSlope[train$LandSlope!='Gtl']<-0


nhbdprice<-summarise(group_by(train,Neighborhood),mean(SalePrice,na.rm=T))
nhbdprice[order(nhbdprice$`mean(SalePrice, na.rm = T)`,decreasing = T),]
train$nhbd_price_lvl[train$Neighborhood %in% filter(nhbdprice,nhbdprice$`mean(SalePrice, na.rm = T)`<140000)$Neighborhood]<-1
train$nhbd_price_lvl[train$Neighborhood %in% filter(nhbdprice,nhbdprice$`mean(SalePrice, na.rm = T)`>=140000 & nhbdprice$`mean(SalePrice, na.rm = T)`<200000)$Neighborhood]<-2
train$nhbd_price_lvl[train$Neighborhood %in% filter(nhbdprice,nhbdprice$`mean(SalePrice, na.rm = T)`>=200000)$Neighborhood]<-3

# summarize(group_by(train,Condition1),mean(SalePrice,na.rm=T))
train$pos_ftr_1[train$Condition1 %in% c('PosA','PosN')]<-1
train$pos_ftr_1[!train$Condition1 %in% c('PosA','PosN')]<-0

# summarize(group_by(train,Condition2),mean(SalePrice,na.rm=T))
train$pos_ftr_1[train$Condition2 %in% c('PosA','PosN')]<-1
train$pos_ftr_1[!train$Condition2 %in% c('PosA','PosN')]<-0

# summarize(group_by(train,BldgType),mean(SalePrice,na.rm=T))
train$BldgType_Mod[train$BldgType %in% c('1Fam','TwnhsE')]<-1
train$BldgType_Mod[!train$BldgType %in% c('1Fam','TwnhsE')]<-0

# summarize(group_by(train,HouseStyle),mean(SalePrice,na.rm=T))
train$HouseStyle_Mod[train$HouseStyle %in% c('2.5Fin','2Story')]<-1
train$HouseStyle_Mod[!train$HouseStyle %in% c('2.5Fin','2Story')]<-0

# summarize(group_by(train,RoofStyle),mean(SalePrice,na.rm=T))
train$RoofStyle_Mod[train$RoofStyle %in% c('Hip','Shed')]<-1
train$RoofStyle_Mod[!train$RoofStyle %in% c('Hip','Shed')]<-0

# summarize(group_by(train,RoofMatl),mean(SalePrice,na.rm=T))
train$RoofMatl_Mod[train$RoofMatl %in% c('Membran','WdShake','WdShngl')]<-1
train$RoofMatl_Mod[!train$RoofMatl %in% c('Membran','WdShake','WdShngl')]<-0

# summarize(group_by(train,Exterior1st),mean(SalePrice,na.rm=T))
train$Exterior1st_mod[train$Exterior1st %in% c('CemntBd','ImStucc','Stone','VinylSd','BrkFace')]<-1
train$Exterior1st_mod[!train$Exterior1st %in% c('CemntBd','ImStucc','Stone','VinylSd','BrkFace')]<-0

# summarize(group_by(train,Exterior2nd),mean(SalePrice,na.rm=T))
train$Exterior2nd_Mod[train$Exterior2nd %in% c('BrkFace','CmentBd','ImStucc','Other','VinylSd')]<-1
train$Exterior2nd_Mod[!train$Exterior2nd %in% c('BrkFace','CmentBd','ImStucc','Other','VinylSd')]<-0

# summarize(group_by(train,MasVnrType),mean(SalePrice,na.rm=T))
train$MasType_Mod[train$MasVnrType %in% c('BrkFace','Stone')]<-1
train$MasType_Mod[!train$MasVnrType %in% c('BrkFace','Stone')]<-0

# summarize(group_by(train,ExterQual),mean(SalePrice,na.rm=T))
train$ExterQual_mod[train$ExterQual=='Ex']<-4
train$ExterQual_mod[train$ExterQual=='Gd']<-3
train$ExterQual_mod[train$ExterQual=='TA']<-2
train$ExterQual_mod[train$ExterQual=='Fa']<-1

# summarize(group_by(train,ExterCond),mean(SalePrice,na.rm=T))
train$ExterCond_mod[train$ExterCond=='Ex']<-5
train$ExterCond_mod[train$ExterCond=='Gd']<-4
train$ExterCond_mod[train$ExterCond=='TA']<-3
train$ExterCond_mod[train$ExterCond=='Fa']<-2
train$ExterCond_mod[train$ExterCond=='Po']<-1

# summarize(group_by(train,Foundation),mean(SalePrice,na.rm=T))
train$Foundation_mod[train$Foundation %in% c('PConc','Stone','Wood')]<-1
train$Foundation_mod[!train$Foundation %in% c('PConc','Stone','Wood')]<-0

train$BsmtQual_mod[train$BsmtQual=='Ex']<-5
train$BsmtQual_mod[train$BsmtQual=='Gd']<-4
train$BsmtQual_mod[train$BsmtQual=='TA']<-3
train$BsmtQual_mod[train$BsmtQual=='Fa']<-2
train$BsmtQual_mod[is.na(train$BsmtQual)]<-1

train$BsmtCond_mod[train$BsmtCond=='Gd']<-5
train$BsmtCond_mod[train$BsmtCond=='TA']<-4
train$BsmtCond_mod[train$BsmtCond=='Fa']<-3
train$BsmtCond_mod[is.na(train$BsmtCond)]<-2
train$BsmtCond_mod[train$BsmtCond=='Po']<-1

# summarize(group_by(train,BsmtExposure),mean(SalePrice,na.rm=T))
train$BsmtExposure_mod[train$BsmtExposure=='Gd']<-5
train$BsmtExposure_mod[train$BsmtExposure=='Av']<-4
train$BsmtExposure_mod[train$BsmtExposure=='Mn']<-3
train$BsmtExposure_mod[train$BsmtExposure=='No']<-2
train$BsmtExposure_mod[is.na(train$BsmtExposure)]<-1

# summarize(group_by(train,BsmtFinType1),mean(SalePrice,na.rm=T))
train$BsmtFinType1_mod[train$BsmtFinType1=='GLQ']<-5
train$BsmtFinType1_mod[train$BsmtFinType1=='Unf']<-4
train$BsmtFinType1_mod[train$BsmtFinType1=='ALQ']<-3
train$BsmtFinType1_mod[train$BsmtFinType1 %in% c('BLQ','LwQ','Rec')]<-1
train$BsmtFinType1_mod[is.na(train$BsmtFinType1)]<-1

# summarize(group_by(train,BsmtFinType2),mean(SalePrice,na.rm=T))
train$BsmtFinType2_mod[train$BsmtFinType2=='ALQ']<-6
train$BsmtFinType2_mod[train$BsmtFinType2=='Unf']<-5
train$BsmtFinType2_mod[train$BsmtFinType2=='GLQ']<-4
train$BsmtFinType2_mod[train$BsmtFinType2 %in% c('LwQ','Rec')]<-3
train$BsmtFinType2_mod[train$BsmtFinType2=='BLQ']<-2
train$BsmtFinType2_mod[is.na(train$BsmtFinType2)]<-1

# summarize(group_by(train,Heating),mean(SalePrice,na.rm=T))
train$Heating_mod[train$Heating=='GasA']<-5
train$Heating_mod[train$Heating=='GasW']<-4
train$Heating_mod[train$Heating=='OthW']<-3
train$Heating_mod[train$Heating=='Wall']<-2
train$Heating_mod[train$Heating %in% c('Floor','Grav')]<-1

# summarize(group_by(train,HeatingQC),mean(SalePrice,na.rm=T))
train$HeatingQC_mod[train$HeatingQC=='Ex']<-5
train$HeatingQC_mod[train$HeatingQC=='Gd']<-4
train$HeatingQC_mod[train$HeatingQC=='TA']<-3
train$HeatingQC_mod[train$HeatingQC=='Fa']<-2
train$HeatingQC_mod[train$HeatingQC=='Po']<-1

train$CentralAir_mod[train$CentralAir=='Y']<-1
train$CentralAir_mod[train$CentralAir!='Y']<-0

# summarize(group_by(train,Electrical),mean(SalePrice,na.rm=T))
train$Electrical_mod[train$Electrical=='SBrkr']<-6
train$Electrical_mod[is.na(train$Electrical)]<-5
train$Electrical_mod[train$Electrical=='FuseA']<-4
train$Electrical_mod[train$Electrical=='FuseF']<-3
train$Electrical_mod[train$Electrical=='FuseP']<-2
train$Electrical_mod[train$Electrical=='Mix']<-1

# summarize(group_by(train,KitchenQual),mean(SalePrice,na.rm=T))
train$KitchenQual_mod[train$KitchenQual=='Ex']<-4
train$KitchenQual_mod[train$KitchenQual=='Gd']<-3
train$KitchenQual_mod[train$KitchenQual=='TA']<-2
train$KitchenQual_mod[train$KitchenQual=='Fa']<-1

# summarize(group_by(train,Functional),mean(SalePrice,na.rm=T))
train$Functional_mod[train$Functional=='Typ']<-6
train$Functional_mod[train$Functional=='Mod']<-5
train$Functional_mod[train$Functional=='Maj1']<-4
train$Functional_mod[train$Functional %in% c('Min1','Min2')]<-3
train$Functional_mod[train$Functional=='Sev']<-2
train$Functional_mod[train$Functional=='Maj2']<-1

# summarize(group_by(train,FireplaceQu),mean(SalePrice,na.rm=T))
train$FireplaceQu_mod[train$FireplaceQu=='Ex']<-6
train$FireplaceQu_mod[train$FireplaceQu=='Gd']<-5
train$FireplaceQu_mod[train$FireplaceQu=='TA']<-4
train$FireplaceQu_mod[train$FireplaceQu=='Fa']<-3
train$FireplaceQu_mod[is.na(train$FireplaceQu)]<-2
train$FireplaceQu_mod[train$FireplaceQu=='Po']<-1

# summarize(group_by(train,GarageType),mean(SalePrice,na.rm=T))
train$GarageType_mod[train$GarageType=='BuiltIn']<-7
train$GarageType_mod[train$GarageType=='Attchd']<-6
train$GarageType_mod[train$GarageType=='Basment']<-5
train$GarageType_mod[train$GarageType=='2Types']<-4
train$GarageType_mod[train$GarageType=='Detchd']<-3
train$GarageType_mod[train$GarageType=='CarPort']<-2
train$GarageType_mod[is.na(train$GarageType)]<-1

# summarize(group_by(train,GarageFinish),mean(SalePrice,na.rm=T))
train$GarageFinish_mod[train$GarageFinish=='Fin']<-4
train$GarageFinish_mod[train$GarageFinish=='RFn']<-3
train$GarageFinish_mod[train$GarageFinish=='Unf']<-2
train$GarageFinish_mod[is.na(train$GarageFinish)]<-1

# summarize(group_by(train,GarageQual),mean(SalePrice,na.rm=T))
train$GarageQual_mod[train$GarageQual=='Ex']<-6
train$GarageQual_mod[train$GarageQual=='Gd']<-5
train$GarageQual_mod[train$GarageQual=='TA']<-4
train$GarageQual_mod[train$GarageQual=='Fa']<-3
train$GarageQual_mod[is.na(train$GarageQual)]<-2
train$GarageQual_mod[train$GarageQual=='Po']<-1

# summarize(group_by(train,GarageCond),mean(SalePrice,na.rm=T))
train$GarageCond_mod[train$GarageCond=='TA']<-6
train$GarageCond_mod[train$GarageCond=='Gd']<-5
train$GarageCond_mod[train$GarageCond=='Ex']<-4
train$GarageCond_mod[train$GarageCond=='Fa']<-3
train$GarageCond_mod[train$GarageCond=='Po']<-2
train$GarageCond_mod[is.na(train$GarageCond)]<-1

# summarise(group_by(train,PavedDrive),mean(SalePrice,na.rm=T))
train$drive_paved[train$PavedDrive=='Y']<-1
train$drive_paved[train$PavedDrive!='Y']<-0
train$drive_paved[is.na(train$PavedDrive)]<-0

# summarise(group_by(train,PoolQC),mean(SalePrice,na.rm=T))
train$PoolQC_mod[train$PoolQC=='Ex']<-1
train$PoolQC_mod[train$PoolQC!='Ex']<-0
train$PoolQC_mod[is.na(train$PoolQC)]<-0

# summarise(group_by(train,Fence),mean(SalePrice,na.rm=T))
train$Fence_mod[train$Fence=='GdPrv']<-1
train$Fence_mod[train$Fence!='GdPrv']<-0
train$Fence_mod[is.na(train$Fence)]<-0

# summarise(group_by(train,SaleType),mean(SalePrice,na.rm=T))
train$SaleType_mod[train$SaleType %in% c('New','Con')]<-5
train$SaleType_mod[train$SaleType %in% c('CWD','ConLI')]<-4
train$SaleType_mod[train$SaleType %in% c('WD')]<-3
train$SaleType_mod[train$SaleType %in% c('COD','ConLw','ConLD')]<-2
train$SaleType_mod[train$SaleType %in% c('Oth')]<-1

# summarise(group_by(train,SaleCondition),mean(SalePrice,na.rm=T))
train$SaleCon_mod[train$SaleCondition=='Partial']<-5
train$SaleCon_mod[train$SaleCondition=='Normal']<-4
train$SaleCon_mod[train$SaleCondition=='Alloca']<-3
train$SaleCon_mod[train$SaleCondition %in% c('Family','Abnorml')]<-2
train$SaleCon_mod[train$SaleCondition=='AdjLand']<-1

# have all the features incase they are needed for furthur manipulation
train$LotShape<-NULL
train$LandContour<-NULL
train$LotConfig<-NULL
train$LandSlope<-NULL
# train$Neighborhood<-NULL
train$Condition1<-NULL
train$Condition2<-NULL
train$BldgType<-NULL
train$HouseStyle<-NULL
train$RoofStyle<-NULL
train$RoofMatl<-NULL
train$Exterior1st<-NULL
train$Exterior2nd<-NULL
train$MasVnrType<-NULL
train$ExterQual<-NULL
train$ExterCond<-NULL
train$Foundation<-NULL
train$BsmtQual<-NULL
train$BsmtCond<-NULL
train$BsmtExposure<-NULL
train$BsmtFinType1<-NULL
train$BsmtFinType2<-NULL
train$Heating<-NULL
train$HeatingQC<-NULL
train$CentralAir<-NULL
train$Electrical<-NULL
train$KitchenQual<-NULL
train$Functional<-NULL
train$FireplaceQu<-NULL
train$GarageType<-NULL
train$PavedDrive<-NULL
train$Fence<-NULL
train$SaleType<-NULL
train$SaleCondition<-NULL
train$PoolQC<-NULL
train$MiscFeature<-NULL
train$Alley<-NULL
train$GarageCond<-NULL
train$GarageQual<-NULL
train$GarageFinish<-NULL
train$Street<-NULL
train$Utilities<-NULL
train$MSZoning<-NULL

names(train)

train$LotFrontage[is.na(train$LotFrontage)]<-0
train$MasVnrArea[is.na(train$MasVnrArea)]<-0
train$GarageYrBlt[is.na(train$GarageYrBlt)]<-0
train[is.na(train)] <- 0
colSums(is.na(train))
sum(is.na(train))

#Based on correlation
train$year_qual <- train$YearBuilt*train$OverallQual #overall condition
train$year_r_qual <- train$YearRemodAdd*train$OverallQual #quality x remodel
train$qual_bsmt <- train$OverallQual*train$TotalBsmtSF #quality x basement size

train$livarea_qual <- train$OverallQual*train$GrLivArea #quality x living area
train$qual_bath <- train$OverallQual*train$FullBath #quality x baths
train$qual_ext <- train$OverallQual*train$Exterior1st_mod #quality x exterior

train$OverallGrade<-train$OverallCond*train$OverallQual
train$GarageGrade<-train$GarageCond * train$GarageQual
train$ExterGrade<-train$ExterCond * train$ExterQual
train$KitchenScore<-train$KitchenAbvGr * train$KitchenQual
train$FireplaceScore <- train$FireplaceQu * train$Fireplaces
train$GarageScore <- train$GarageArea * train$GarageQual
train$PoolScore <- train$PoolArea * train$PoolQC
train$TotalBath <- train$BsmtFullBath + train$BsmtHalfBath*0.5 + train$FullBath + train$HalfBath*0.5
train$AllSF<- train$GrLivArea + train$TotalBsmtSF
train$AllFlrsSF<- train$X1stFlrSF + train$X2ndFlrSF
train$AllPorchSF<- train$OpenPorchSF + train$X3SsnPorch + train$ScreenPorch + train$EnclosedPorch

#replace LotFrontage with median
train$LotFrontage[train$LotFrontage==0]<-NA
# train$LotFrontage[is.na(train$LotFrontage)]<-median(train$LotFrontage,na.rm=TRUE)
#summarise(group_by(train1 ,Neighborhood),median(LotFrontage,na.rm=T))

names(train)

write.csv(train,"C:/Users/vchandramohan/Desktop/Vyshali/House Prediction/train_py_clean.csv",row.names = F)

