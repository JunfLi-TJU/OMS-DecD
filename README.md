Source codes of algorithms and datasets for our paper "On the Necessity of Collaboration for
Online Model Selection with Decentralized Data", accepted in NeurIPS 2024.

We implement all algorithms with R on a Windows machine with 2.8 GHz Core(TM) i7-1165G7 CPU, execute each experiment 10 times with random permutation of all datasets and average all of the results.

The default path of codes is "D:/experiment/NeurIPS/NeurIPS2024/code/".

The path of datasets is "D:/experiment/online learning dataset/regression/".

The store path is "D:/experiment/NeurIPS/NeurIPS2024/Result/".

You can also change all of the default paths.

The baseline algorithms include: eM-KOFL and POF-MKL. 
Our algorithms include: AOMD-OGS-hinge, AOMD-OGS-square, AONS-OGS, AVP-OGS, MAVP-OGS.

The datasets are downloaded from: https://archive.ics.uci.edu/ml/index.php
and 
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html.

AOMD-OGS-hinge is obtained by instantiating AOMD-OGS with the Hinge loss function.
AOMD-OGS-square is obtained by instantiating AOMD-OGS with the square loss function.
FOGD-hinge is obtained by instantiating FOGD with the Hinge loss function.
