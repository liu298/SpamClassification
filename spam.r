setwd('/Users/yolanda/Box\ Sync/DataScience/Project/SpamClassification/machine-learning-ex6/ex6')
library(R.matlab)
library(kernlab)
library(caret)
library(glmnet)

# get training and test data set
train = as.matrix(readMat('spamTrain.mat',fixNames = TRUE))
train.x = as.data.frame(train[[1]])
# train.x = as.data.frame(apply(train.x,2,factor))
train.y = as.vector(train[[2]])
test = as.matrix(readMat('spamTest.mat',fixNames = TRUE))
test.x = as.data.frame(test[[1]])
# test.x = as.data.frame(apply(test.x,2,factor))
test.y = as.vector(test[[2]])
dim(train.x);
mean(train.y);

# Model Evaluation
eva = function(pred,y){
  mat = table(pred,y)
  tp = mat[1,1]
  tn = mat[2,2]
  fn = mat[1,2]
  fp = mat[2,1]
  prec = tp / (tp+fp)
  recall = tp / (tp+fn)
  specificity = tn / (tn+fn)
  sensitivity = tp / (tp+fp)
  f1 = 2*prec*recall / (prec+recall)
  accuracy = (tp+tn)/(tp+tn+fn+fp)
  c(accuracy,prec,recall,sensitivity,specificity,f1)
}

roc <- function(y, pred) {
  N = length(y)
  alpha <- quantile(pred, seq(0,1,length.out = N))
  
  sens <- rep(NA,N)
  spec <- rep(NA,N)
  for (i in 1:N) {
    predClass <- as.numeric(pred >= alpha[i])
    sens[i] <- sum(predClass == 1 & y == 1) / sum(y == 1)
    spec[i] <- sum(predClass == 0 & y == 0) / sum(y == 0)
  }
  return(list(fpr=1 - spec, tpr=sens))
}

auc <- function(r) {
  sum((r$fpr) * diff(c(0,r$tpr)))
}


summary(as.factor(train.y)) # non-spam 0.681
# modification for unbalanced

# linear kernel
cost = .1
linSVM1 = ksvm(train.y~.,data=train.x,type="C-svc",kernel='vanilladot',C=cost,scaled=c())
lin1 = predict(linSVM1,test.x)
eva(lin1,test.y)

# ROC
cost = 1
linSVM2 = ksvm(train.y~.,data=train.x,type="C-svc",kernel='vanilladot',C=cost,scaled=c())
lin2 = predict(linSVM2,test.x)
eva(lin2,test.y)

# linear kernel cv tuning with caret
svmGrid = expand.grid(C=c(2^(-3:3)),degree=c(1,2),scale=FALSE)
svmControl = trainControl(method="cv",number=5)
linSVM = train(train.y~.,data=train.x,method="svmPoly",
                trControl=svmControl,tuneGrid=svmGrid)
lin.pred = predict(linSVM,test.x)
linSVM$bestTune;

ctrl <- trainControl(method="cv",   # 5fold cross validation
                     number=5,		    
                     summaryFunction=twoClassSummary,	# Use AUC to pick the best model
                     classProbs=TRUE)
#Train and Tune the SVM
class = rep(NA,length(train.y))
class[train.y==1] = "spam"
class[train.y!=1] = "non"
svm.tune2 <- train(x=train.x,
                   y= class,
                   method = "svmLinear",
                   preProc = c("center","scale"),
                   metric="ROC",
                   trControl=ctrl)

# logistic regression
glm.fit = glm(train.y~.,data=train.x,na.action=na.omit,family = binomial)
glm.pred = predict(glm.fit,test.x,type="response")
log.pred = ifelse(glm.pred>=0.7,1,0)
eva(log.pred, test.y)


# logistic regression with elastic net
a = seq(0, 1, 0.1)
lambda = rep(NA, 11)
elnetPredMat = matrix(NA, length(test.y), 11)
evaMat = matrix(NA, 6, 11)
auc_all = rep(NA,11)
for (i in 1:11){
  alpha = a[i]
  elnet.cv = cv.glmnet(as.matrix(train.x),train.y,alpha = alpha,nfolds = 5,family = 'binomial',type.measure='auc')
  elnet.cv2 = glmnet(as.matrix(train.x),train.y,alpha = alpha, lambda = elnet.cv$lambda.min,family = 'binomial')
  lambda[i] = elnet.cv$lambda.min
  yhat = predict(elnet.cv2,as.matrix(test.x),type="response")
  auc_all[i] = auc(roc(test.y,yhat))
  elnetPredMat[,i] = yhat>=0.65
  (temp = eva(yhat,test.y))
  evaMat[,i] = as.vector(temp)
}

# Elastic Net: select best alpha.
alphaall <- seq(0, 1, length.out = 10)
auc_all <- c()
lambda_all <- c()
for (i in 1:length(alphaall)) {
  alpha <- alphaall[i]
  outLasso <- glmnet(as.matrix(train.x), train.y, family="binomial", alpha = alpha)
  aucValsLasso <- rep(NA, length(outLasso$lambda))
  for (j in 1:length(outLasso$lambda)) {
    pred <- 1 / (1 + exp(-1 * as.matrix(cbind(1, test.x)) %*% coef(outLasso)[,j]))
    r <- roc(test.y, pred)
    plot(r$fpr,r$tpr)
    if(auc(r)>1){
      print(c(j,r))
    }
    aucValsLasso[j] <- auc(r)
  }
  best <- which.max(aucValsLasso)
  auc_all <- c(auc_all, aucValsLasso[best])
  lambda_all <- c(lambda_all, outLasso$lambda[best])
}

evaMat
lambda
a
elnet.cv = glmnet(as.matrix(train.x),train.y,alpha = 0.1, lambda = 0.0032,family = 'binomial')
elnet = predict(elnet.cv,as.matrix(test.x))>=0.7
elnet.probs = predict(elnet.cv,as.matrix(test.x))
auc(roc(test.y, elnet.probs)) # 0.997
ensemble = apply(as.matrix(cbind(lin1,lin2,elnet)),1,mean)>=0.5
eva(ensemble,test.y)
eva(elnet,test.y)

# stimulation
set.seed(322222)
x = matrix(sample(c(0,1),100,replace = TRUE),10,10)
x = as.data.frame(apply(x,2,factor))
y = as.factor(sample(c(0,1),10,replace = TRUE))

head(x)
set.seed(21)
tx = matrix(sample(c(0,1),100,replace = TRUE),10,10)
tx = as.data.frame(apply(x,2,factor))
glm.sim = glm(y~.,data=x,family = binomial(link = "logit"))
pred = predict(glm.sim,tx,type = 'response')>=0.5
table(pred,y)


eva(glm.pred,test.y)
eva(lin1,test.y)
