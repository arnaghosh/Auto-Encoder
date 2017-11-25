# How to run :-
+ [advAE_cnn_mnist.lua](https://github.com/arnaghosh/Auto-Encoder/blob/master/Cluster%20Visualisation/advAE_cnn_mnist.lua) - Run this if adversary not saved. This is adhoc. Preferable to use the adversary. This saves another network with same structure as adversary trained on discriminating the classes from dictionary codes.
+ [calc_imp_bits.lua](https://github.com/arnaghosh/Auto-Encoder/blob/master/Cluster%20Visualisation/calc_imp_bits.lua) - This calculates the importance of each bit by forwarding the inputs with a bit masked and seeing hwo classification error changes. Masking the more important bit will significantly increase error. 
+ [convert_dict_to_features.lua](https://github.com/arnaghosh/Auto-Encoder/blob/master/Cluster%20Visualisation/convert_dict_to_features.lua) - The bits are split into 2 halves - alternate ones form the basis for each feature (in 2D space). So, the top 2 most important bits form the MSB of the 2 features. This creates the transformation matrix for dictionary.
+ [visualiseCluster.m](https://github.com/arnaghosh/Auto-Encoder/blob/master/Cluster%20Visualisation/visualiseCluster.m) - Converts the dictionary to 2D feature space and generates a scatter plot with each class with a unique color. Also generates a figure showing the color label of each class.

Hope this helps visualise clusters in 2D space for any dataset.
