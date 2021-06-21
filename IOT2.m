clc
clear all
close all

%% Train Data
DATATR1 = fullfile('images\Train');
imdsTR = imageDatastore(DATATR1,'IncludeSubfolders',true,'LabelSource','foldernames');

%% Test Data
DATATS1 = fullfile('images\Test');
imdsTS = imageDatastore(DATATS1,'IncludeSubfolders',true,'LabelSource','foldernames');

%% 
labelCount1 = countEachLabel(imdsTR);
labelCount2 = countEachLabel(imdsTS);

%% 4
outputSize=[224 224];

% net = alexnet;g
net = googlenet;
% net = squeezenet;
% net = vgg19;

net.Layers
%%
analyzeNetwork(net)
%%
auimdstrain = augmentedImageDatastore(outputSize,imdsTR);
auimdstest = augmentedImageDatastore(outputSize,imdsTS);
layer = 'pool5-drop_7x7_s1';
featuresTrain = activations(net,auimdstrain,layer,'OutputAs','rows');
% featuresTest = activations(net,auimdstest,layer,'OutputAs','rows');

Feature = featuresTrain;

xTrain = Feature(1:1000 , 1:2:end);
xTest = Feature(1001:end , 1:2:end);
%% 5

Tr = xlsread('train.csv');
% Te = xlsread('test.csv');
Target = Tr;

Tr = Target(1:1000 , 1);
Te = Target(1001:end , 1);

YTrain = Tr;
YTest = Te;
%% 6

%% 9. SVM

svmMd = fitcecoc(xTrain,YTrain); % Fit multiclass models for support vector machines or other classifiers
svmPredicate = predict(svmMd,xTest);

figure;
plot(svmPredicate)
hold on
plot(YTest)
grid minor
legend('SVM','Target')

MSE_SVM = 100*(mean(abs(svmPredicate - YTest)))

% plotconfusion(YTest,svmPredicate)
% [Se, Sp, Ppv, Npv, Acc, TP, TN, FP, FN, Precision, Recall,F1Score] =
% performance_measure(svmPredicate, YTest);

%% 12. Ensemble

ensembles = fitcensemble(xTrain,YTrain); % Fit ensemble of learners for classification
enPredicate = predict(ensembles,xTest);

figure;
plot(enPredicate)
hold on
plot(YTest)
grid minor
legend('Ensemble','Target')

MSE_Ensemble = 100*(mean(abs(enPredicate - YTest)))

% plotconfusion(YTest,enPredicate)

%% 14. Discriminant

Discriminantclass = fitcdiscr(xTrain,YTrain); % Fit discriminant analysis classifier
discrPredicate = predict(Discriminantclass,xTest);

figure;
plot(discrPredicate)
hold on
plot(YTest)
grid minor
legend('Discriminant','Target')

MSE_discrPredicate = 100*(mean(abs(discrPredicate - YTest)))
