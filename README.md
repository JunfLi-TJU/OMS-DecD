Source codes of algorithms and datasets for our paper "On the Necessity of Collaboration for
Online Model Selection with Decentralized Data", accepted in NeurIPS 2024.

We implement all algorithms with R on a Windows machine with 2.8 GHz Core(TM) i7-1165G7 CPU, execute each experiment 10 times with random permutation of all datasets and average all of the results.

The default path of codes is "D:/experiment/NeurIPS2024/code".

The path of datasets is "D:/experiment/NeurIPS2024/dataset".

The store path is "D:/experiment/NeurIPS2024/Result/".

You can also change all of the default paths.

The baseline algorithms include: eM-KOFL and POF-MKL. 

The datasets are downloaded from WEKA: https://waikato.github.io/weka-wiki/datasets/
and UCI machine learning repository: https://archive.ics.uci.edu/ml/index.php.

For Table 3, 
please run FOMD-OMS-linear-2, FOMD-OMS-linear-K, NCO-OMS-2 and NCO-OMS-K.
FOMD-OMS-linear-2 is obtained by running FOMD-OMS with J=2.
NCO-OMS-2 is obtained by running NCO-OMS with J=2.

For Table 4,
please run eM-KOFL, POF-MKL and FOMD-OMS.

