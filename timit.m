%% Data Input

fs = 16000;

data_rm1 = wavread('../data/TIMIT/Rag/Male/Speaker1.WAV');
data_rm2 = wavread('../data/TIMIT/Rag/Male/Speaker2.WAV');
data_rm3 = wavread('../data/TIMIT/Rag/Male/Speaker3.WAV');
data_rm4 = wavread('../data/TIMIT/Rag/Male/Speaker4.WAV');
data_rf1 = wavread('../data/TIMIT/Rag/Female/Speaker1.WAV');
data_rf2 = wavread('../data/TIMIT/Rag/Female/Speaker2.WAV');
data_rf3 = wavread('../data/TIMIT/Rag/Female/Speaker3.WAV');
data_rf4 = wavread('../data/TIMIT/Rag/Female/Speaker4.WAV');
data_sm1 = wavread('../data/TIMIT/Suit/Male/Speaker1.WAV');
data_sm2 = wavread('../data/TIMIT/Suit/Male/Speaker2.WAV');
data_sm3 = wavread('../data/TIMIT/Suit/Male/Speaker3.WAV');
data_sm4 = wavread('../data/TIMIT/Suit/Male/Speaker4.WAV');
data_sf1 = wavread('../data/TIMIT/Suit/Female/Speaker1.WAV');
data_sf2 = wavread('../data/TIMIT/Suit/Female/Speaker2.WAV');
data_sf3 = wavread('../data/TIMIT/Suit/Female/Speaker3.WAV');
data_sf4 = wavread('../data/TIMIT/Suit/Female/Speaker4.WAV');

training_data1 = data_rm1;
training_data2 = data_rm2;
training_data3 = data_rm3;
training_data4 = data_rm4;
training_data5 = data_rf1;
training_data6 = data_rf2;
training_data7 = data_rf3;
training_data8 = data_rf4;

testing_data1 = data_sm1;
testing_data2 = data_sm2;
testing_data3 = data_sm3;
testing_data4 = data_sm4;
testing_data5 = data_sf1;
testing_data6 = data_sf2;
testing_data7 = data_sf3;
testing_data8 = data_sf4;

%%

%% Feature Extraction (MFCCs)

training_features1 = melcepst(training_data1, fs);
training_features2 = melcepst(training_data2, fs);
training_features3 = melcepst(training_data3, fs);
training_features4 = melcepst(training_data4, fs);
training_features5 = melcepst(training_data5, fs);
training_features6 = melcepst(training_data6, fs);
training_features7 = melcepst(training_data7, fs);
training_features8 = melcepst(training_data8, fs);

testing_features1  = melcepst(testing_data1, fs);
testing_features2  = melcepst(testing_data2, fs);
testing_features3  = melcepst(testing_data3, fs);
testing_features4  = melcepst(testing_data4, fs);
testing_features5  = melcepst(testing_data5, fs);
testing_features6  = melcepst(testing_data6, fs);
testing_features7  = melcepst(testing_data7, fs);
testing_features8  = melcepst(testing_data8, fs);

%% Get some interesting graphs

% subplot(2,1,1),
a = melcepst(training_data1, fs);
% subplot(2,1,2),
% a = melcepst(training_data5, fs);


%% Feature Matching (Minimum-Distance Classifier)

delta = 0.88; % threshold for identification

mean_training_feature1 = mean(training_features1);
mean_training_feature2 = mean(training_features2);
mean_training_feature3 = mean(training_features3);
mean_training_feature4 = mean(training_features4);
mean_training_feature5 = mean(training_features5);
mean_training_feature6 = mean(training_features6);
mean_training_feature7 = mean(training_features7);
mean_training_feature8 = mean(training_features8);

mean_testing_feature1 = mean(testing_features1);
mean_testing_feature2 = mean(testing_features2);
mean_testing_feature3 = mean(testing_features3);
mean_testing_feature4 = mean(testing_features4);
mean_testing_feature5 = mean(testing_features5);
mean_testing_feature6 = mean(testing_features6);
mean_testing_feature7 = mean(testing_features7);
mean_testing_feature8 = mean(testing_features8);

d11 = mean((mean_training_feature1 - mean_testing_feature1).^2);
d12 = mean((mean_training_feature1 - mean_testing_feature2).^2);
d13 = mean((mean_training_feature1 - mean_testing_feature3).^2);
d14 = mean((mean_training_feature1 - mean_testing_feature4).^2);
d15 = mean((mean_training_feature1 - mean_testing_feature5).^2);
d16 = mean((mean_training_feature1 - mean_testing_feature6).^2);
d17 = mean((mean_training_feature1 - mean_testing_feature7).^2);
d18 = mean((mean_training_feature1 - mean_testing_feature8).^2);

d21 = mean((mean_training_feature2 - mean_testing_feature1).^2);
d22 = mean((mean_training_feature2 - mean_testing_feature2).^2);
d23 = mean((mean_training_feature2 - mean_testing_feature3).^2);
d24 = mean((mean_training_feature2 - mean_testing_feature4).^2);
d25 = mean((mean_training_feature2 - mean_testing_feature5).^2);
d26 = mean((mean_training_feature2 - mean_testing_feature6).^2);
d27 = mean((mean_training_feature2 - mean_testing_feature7).^2);
d28 = mean((mean_training_feature2 - mean_testing_feature8).^2);

d31 = mean((mean_training_feature3 - mean_testing_feature1).^2);
d32 = mean((mean_training_feature3 - mean_testing_feature2).^2);
d33 = mean((mean_training_feature3 - mean_testing_feature3).^2);
d34 = mean((mean_training_feature3 - mean_testing_feature4).^2);
d35 = mean((mean_training_feature3 - mean_testing_feature5).^2);
d36 = mean((mean_training_feature3 - mean_testing_feature6).^2);
d37 = mean((mean_training_feature3 - mean_testing_feature7).^2);
d38 = mean((mean_training_feature3 - mean_testing_feature8).^2);

d41 = mean((mean_training_feature4 - mean_testing_feature1).^2);
d42 = mean((mean_training_feature4 - mean_testing_feature2).^2);
d43 = mean((mean_training_feature4 - mean_testing_feature3).^2);
d44 = mean((mean_training_feature4 - mean_testing_feature4).^2);
d45 = mean((mean_training_feature4 - mean_testing_feature5).^2);
d46 = mean((mean_training_feature4 - mean_testing_feature6).^2);
d47 = mean((mean_training_feature4 - mean_testing_feature7).^2);
d48 = mean((mean_training_feature4 - mean_testing_feature8).^2);

d51 = mean((mean_training_feature5 - mean_testing_feature1).^2);
d52 = mean((mean_training_feature5 - mean_testing_feature2).^2);
d53 = mean((mean_training_feature5 - mean_testing_feature3).^2);
d54 = mean((mean_training_feature5 - mean_testing_feature4).^2);
d55 = mean((mean_training_feature5 - mean_testing_feature5).^2);
d56 = mean((mean_training_feature5 - mean_testing_feature6).^2);
d57 = mean((mean_training_feature5 - mean_testing_feature7).^2);
d58 = mean((mean_training_feature5 - mean_testing_feature8).^2);

d61 = mean((mean_training_feature6 - mean_testing_feature1).^2);
d62 = mean((mean_training_feature6 - mean_testing_feature2).^2);
d63 = mean((mean_training_feature6 - mean_testing_feature3).^2);
d64 = mean((mean_training_feature6 - mean_testing_feature4).^2);
d65 = mean((mean_training_feature6 - mean_testing_feature5).^2);
d66 = mean((mean_training_feature6 - mean_testing_feature6).^2);
d67 = mean((mean_training_feature6 - mean_testing_feature7).^2);
d68 = mean((mean_training_feature6 - mean_testing_feature8).^2);

d71 = mean((mean_training_feature7 - mean_testing_feature1).^2);
d72 = mean((mean_training_feature7 - mean_testing_feature2).^2);
d73 = mean((mean_training_feature7 - mean_testing_feature3).^2);
d74 = mean((mean_training_feature7 - mean_testing_feature4).^2);
d75 = mean((mean_training_feature7 - mean_testing_feature5).^2);
d76 = mean((mean_training_feature7 - mean_testing_feature6).^2);
d77 = mean((mean_training_feature7 - mean_testing_feature7).^2);
d78 = mean((mean_training_feature7 - mean_testing_feature8).^2);

d81 = mean((mean_training_feature8 - mean_testing_feature1).^2);
d82 = mean((mean_training_feature8 - mean_testing_feature2).^2);
d83 = mean((mean_training_feature8 - mean_testing_feature3).^2);
d84 = mean((mean_training_feature8 - mean_testing_feature4).^2);
d85 = mean((mean_training_feature8 - mean_testing_feature5).^2);
d86 = mean((mean_training_feature8 - mean_testing_feature6).^2);
d87 = mean((mean_training_feature8 - mean_testing_feature7).^2);
d88 = mean((mean_training_feature8 - mean_testing_feature8).^2);

% check_identified(d11, delta, 'd11'); % false negative ; positive
% check_identified(d12, delta, 'd12'); % negative       ; negative
% check_identified(d13, delta, 'd13'); % negative       ; negative
% check_identified(d21, delta, 'd21'); % negative       ; false positive
% check_identified(d22, delta, 'd22'); % positive       ; positive
% check_identified(d23, delta, 'd23'); % false positive ; negative
% check_identified(d31, delta, 'd31'); % negative       ; false positive (negative w delta .88)
% check_identified(d32, delta, 'd32'); % false positive ; negative
% check_identified(d33, delta, 'd33'); % positive       ; positive

% 2/3 correctly identified            ; 3/3 correctly identified
% 1 false negative, 2 false positives ; 2 false positives

check_identified(d11, delta, 'd11');
check_identified(d12, delta, 'd12');
check_identified(d13, delta, 'd13');
check_identified(d14, delta, 'd14');
check_identified(d15, delta, 'd15');
check_identified(d16, delta, 'd16');
check_identified(d17, delta, 'd17');
check_identified(d18, delta, 'd18');

check_identified(d21, delta, 'd21');
check_identified(d22, delta, 'd22');
check_identified(d23, delta, 'd23');
check_identified(d24, delta, 'd24');
check_identified(d25, delta, 'd25');
check_identified(d26, delta, 'd26');
check_identified(d27, delta, 'd27');
check_identified(d28, delta, 'd28');

check_identified(d31, delta, 'd31');
check_identified(d32, delta, 'd32');
check_identified(d33, delta, 'd33');
check_identified(d34, delta, 'd34');
check_identified(d35, delta, 'd35');
check_identified(d36, delta, 'd36');
check_identified(d37, delta, 'd37');
check_identified(d38, delta, 'd38');

check_identified(d41, delta, 'd41');
check_identified(d42, delta, 'd42');
check_identified(d43, delta, 'd43');
check_identified(d44, delta, 'd44');
check_identified(d45, delta, 'd45');
check_identified(d46, delta, 'd46');
check_identified(d47, delta, 'd47');
check_identified(d48, delta, 'd48');

check_identified(d51, delta, 'd51');
check_identified(d52, delta, 'd52');
check_identified(d53, delta, 'd53');
check_identified(d54, delta, 'd54');
check_identified(d55, delta, 'd55');
check_identified(d56, delta, 'd56');
check_identified(d57, delta, 'd57');
check_identified(d58, delta, 'd58');

check_identified(d61, delta, 'd61');
check_identified(d62, delta, 'd62');
check_identified(d63, delta, 'd63');
check_identified(d64, delta, 'd64');
check_identified(d65, delta, 'd65');
check_identified(d66, delta, 'd66');
check_identified(d67, delta, 'd67');
check_identified(d68, delta, 'd68');

check_identified(d71, delta, 'd71');
check_identified(d72, delta, 'd72');
check_identified(d73, delta, 'd73');
check_identified(d74, delta, 'd74');
check_identified(d75, delta, 'd75');
check_identified(d76, delta, 'd76');
check_identified(d77, delta, 'd77');
check_identified(d78, delta, 'd78');

check_identified(d81, delta, 'd81');
check_identified(d82, delta, 'd82');
check_identified(d83, delta, 'd83');
check_identified(d84, delta, 'd84');
check_identified(d85, delta, 'd85');
check_identified(d86, delta, 'd86');
check_identified(d87, delta, 'd87');
check_identified(d88, delta, 'd88');

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
