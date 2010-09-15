%% Data Input

fs = 8000;

training_data1 = wavread('01_train.wav');
training_data2 = wavread('02_train.wav');
training_data3 = wavread('03_train.wav');

testing_data1  = wavread('01_test.wav');
testing_data2  = wavread('02_test.wav');
testing_data3  = wavread('03_test.wav');

%% Feature Extraction (MFCCs)

training_features1 = melcepst(training_data1, fs);
training_features2 = melcepst(training_data2, fs);
training_features3 = melcepst(training_data3, fs);

testing_features1  = melcepst(testing_data1, fs);
testing_features2  = melcepst(testing_data2, fs);
testing_features3  = melcepst(testing_data3, fs);

%% Feature Matching (Minimum-Distance Classifier)

delta = 0.85; % threshold for identification

mean_training_feature1 = mean(training_features1);
mean_training_feature2 = mean(training_features2);
mean_training_feature3 = mean(training_features3);

mean_testing_feature1 = mean(testing_features1);
mean_testing_feature2 = mean(testing_features2);
mean_testing_feature3 = mean(testing_features3);

d11 = mean((mean_training_feature1 - mean_testing_feature1).^2);
d12 = mean((mean_training_feature1 - mean_testing_feature2).^2);
d13 = mean((mean_training_feature1 - mean_testing_feature3).^2);
d21 = mean((mean_training_feature2 - mean_testing_feature1).^2);
d22 = mean((mean_training_feature2 - mean_testing_feature2).^2);
d23 = mean((mean_training_feature2 - mean_testing_feature3).^2);
d31 = mean((mean_training_feature3 - mean_testing_feature1).^2);
d32 = mean((mean_training_feature3 - mean_testing_feature2).^2);
d33 = mean((mean_training_feature3 - mean_testing_feature3).^2);

check_identified(d11, delta, 'd11'); % false negative
check_identified(d12, delta, 'd12'); % negative
check_identified(d13, delta, 'd13'); % negative
check_identified(d21, delta, 'd21'); % negative
check_identified(d22, delta, 'd22'); % positive
check_identified(d23, delta, 'd23'); % false positive
check_identified(d31, delta, 'd31'); % negative
check_identified(d32, delta, 'd32'); % false positive
check_identified(d33, delta, 'd33'); % positive

% 2/3 correctly identified
% 1 false negative, 2 false positives

%% Feature Matching (Gaussian Mixture Model)

No_of_Clusters = 2;
No_of_Iterations = 10;

[training_idx1, training_mu1, training_sigma1] = GMM(training_features1', No_of_Clusters, No_of_Iterations);
[training_idx2, training_mu2, training_sigma2] = GMM(training_features2', No_of_Clusters, No_of_Iterations);
[training_idx3, training_mu3, training_sigma3] = GMM(training_features3', No_of_Clusters, No_of_Iterations);

%%

size(training_features1), size(training_features2), size(training_features3)
size(testing_features1), size(testing_features2), size(testing_features3)
size(training_mu1), size(training_mu2), size(training_mu3)

[pc11, idx11] = Cluster_Probability(testing_features1', training_mu3)
% [pc12, idx12] = Cluster_Probability(testing_features2, training_mu1)
% [pc13, idx13] = Cluster_Probability(testing_features3, training_mu1)
% 
% [pc21, idx21] = Cluster_Probability(testing_features1, training_mu2)
% [pc22, idx22] = Cluster_Probability(testing_features2, training_mu2)
% [pc23, idx23] = Cluster_Probability(testing_features3, training_mu2)
% 
% [pc31, idx31] = Cluster_Probability(testing_features1, training_mu3)
% [pc32, idx32] = Cluster_Probability(testing_features2, training_mu3)
% [pc33, idx33] = Cluster_Probability(testing_features3, training_mu3)

pc11

Cluster_Probability(testing_features1', training_mu1)
Cluster_Probability(testing_features1', training_mu2)
log(Cluster_Probability(testing_features1', training_mu1)) - log(Cluster_Probability(testing_features1', background_mu))
