library(mlbench)
library(dplyr)
library(corrplot)
library(e1071)
library(class)

#view data
data(PimaIndiansDiabetes2,package="mlbench")
raw<-PimaIndiansDiabetes2
dim(raw)
summary(raw)
str(raw)

#modify diabetes values
raw$diabetes<-as.numeric(raw$diabetes)
raw[raw$diabetes=="1", "diabetes"]<-0
raw[raw$diabetes=="2", "diabetes"]<-1

#removing incomplete cases
raw2 <-raw[complete.cases(raw),]
raw3=raw2[1:8]
#visualise correlation between predictor variables
corrplot(cor(raw3),method="color",addCoef.col = "black",number.cex = 1)
#convert diabetes back to factor variable
raw2$diabetes<-as.factor(raw2$diabetes)

#view complete cases
dia=raw2
dim(dia)
str(dia)
summary(dia)
#we can see that we have 262 records for non-diabetic class and 130 records for diabetic class.

#Normalize the numeric variables
nor <-function(x) { (x -min(x))/(max(x)-min(x)) }
dia[,1:8] <- sapply(dia[,1:8], nor)
                    
#split data
set.seed(100)
training.idx <- sample(1: nrow(dia), size=nrow(dia)*0.8)
train.data <-dia[training.idx, ]
test.data <- dia[-training.idx, ]

#logistic regression
set.seed(101)
mlogit <- glm(diabetes~., data = train.data, family = "binomial")
summary(mlogit)

Pred.p <-predict(mlogit, newdata =test.data, type = "response")
y_pred_num <-ifelse(Pred.p > 0.5, 1, 0)
y_pred <-factor(y_pred_num, levels=c(0, 1))

#Accuracy of the classification
mean(y_pred ==test.data$diabetes)
#Create the confusion matrix 
tab <-table(y_pred,test.data$diabetes)
print(tab)


#SVM classification

#SVM classification with linear kernel
set.seed(103)
m.svm<-svm(diabetes~., data=train.data, kernel="linear")
summary(m.svm)

#predict newdata in test set
pred.svm <- predict(m.svm, newdata=test.data[,1:9])
#evaluate classification performance and check accuracy
table(pred.svm, test.data$diabetes)
mean(pred.svm==test.data$diabetes)

#SVM classification with radial kernel
set.seed(103)
m.svm.tune<-tune.svm(diabetes~., data=train.data, kernel="radial", cost=10^(-1:2), gamma=c(.1,.5,1,2))
summary(m.svm.tune)

#confusion matrix and accuracy
best.svm = m.svm.tune$best.model
pred.svm.tune = predict(best.svm, newdata=test.data[,1:9])
table(pred.svm.tune, test.data$diabetes)
mean(pred.svm.tune ==test.data$diabetes)
#classification accuracy with linear kernel is better

#kNN classification
set.seed(104)
#try k=5 first
knn1<-knn(train.data[,1:8], test.data[,1:8], cl=train.data$diabetes, k=5)
mean(knn1 ==test.data$diabetes)
table(knn1,test.data$diabetes)

#Find best value of k for the best accuracy
ac<-rep(0, 30)
for(i in 1:30){
  set.seed(104)
  knn.i<-knn(train.data[,1:8], test.data[,1:8], cl=train.data$diabetes, k=i)
  ac[i]<-mean(knn.i ==test.data$diabetes) 
  cat("k=", i, " accuracy=", ac[i], "\n")
} 

#Accuracy plot
plot(ac, type="b", xlab="K",ylab="Accuracy")
set.seed(105)
knn2<-knn(train.data[,1:8], test.data[,1:8], cl=train.data$diabetes, k=25)
#we choose k=25 because it is an odd number and avoids tied votes
mean(knn2 ==test.data$diabetes) 
#confusion matrix
table(knn2,test.data$diabetes)
