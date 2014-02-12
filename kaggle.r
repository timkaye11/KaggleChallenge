require (randomForest)
require (imputation)
require (e1071)
require (dismo)

yvec <- as.factor(cs.training[,2])

#Imputes with random forest and then creates a random forest to predict
bigData <- rfImpute(y = yvec[1:3000], x = cs.training[1:3000,3:12])
rfiData <- bigData[,2:11]
rfForest <- randomForest(y = yvec[1:3000], x = rfiData, cutoff = c(.5,.5))

# Imputes using generalized boosting and then creates a random forest to predict
# Also, I rescind my previous preference of rfImputation. Boosting performs slightly
# better when analyzing the forest’s ability to explain variance
y.vec <- as.factor(cs.training[,2])
y.sample <- y.vec[1:5000]
bigData <- gbmImpute(x = cs.training[1:5000,3:12], n.trees=950, cv.fold = 10)
gbmData <- data.frame(bigData[1])
gForest <- randomForest(y = as.factor(yvec[1:5000]), x = gbmData, cutoff = c(.5,.5), ntrees = 1000, mtry=sqrt(ncol(gbmData)))

# ! regarding the gForest assignment line, do you get a ‘Are you sure you want to do regression warning message’?
test.data <- read.csv("/Users/timkaye11/Downloads/cs-test.csv")
newSet <- test.data[,3:12]
newImputedTest <- gbmImpute(x=newSet, n.trees=950, cv.fold=10)
newImputedValues <- data.frame(newImputedTest[1])
preds <- predict(gForest, newdata = newImputedValues, type = "prob")
testPreds <- preds[,2]
write.csv(data.frame(testPreds), "tryagain1.csv")


rfForest
gForest

# This code allows for cross validation or prediction once you have a forest
# For example .9 means that the tree makes 9/10 as many errors as guessing that
# no one will default
ghettoSample <- 10001:14000
ghettoData <- gbmImpute(x = cs.training[ghettoSample,3:12])
ghettoData <- data.frame(ghettoData[1])
ghettoPredictions <- predict(gForest, newdata = ghettoData)
diff <- sum(abs(as.numeric(ghettoPredictions) - as.numeric(yvec[ghettoSample])))
percentErrorsVNone <- diff/ sum(as.numeric(yvec[ghettoSample]) - 1)
percentErrorsVNone 