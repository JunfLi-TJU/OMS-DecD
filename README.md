Source codes of algorithms and datasets for our paper "On the Necessity of Collaboration for
Online Model Selection with Decentralized Data", accepted by NeurIPS 2024.

We implement all algorithms with R on a Windows machine with 2.8 GHz Core(TM) i7-1165G7 CPU, execute each experiment 10 times with random permutation of all datasets and average all of the results.

The default path of codes is "D:/experiment/JMLR/JMLR2024/code/".

The path of datasets is "D:/experiment/online learning dataset/regression/".

The path of datasets is "D:/experiment/online learning dataset/binary C/".

The path of datasets is "D:/experiment/online learning dataset/multi-class".

The store path is "D:/experiment/JMLR/JMLR2024/Result/".

You can also change all of the default paths.

The baseline algorithms include: 
Projectron++, M-Projectron++, FOGD-hinge, M-FOGD, PROS-N-KONS and CON-KONS. 
Our algorithms include: AOMD-OGS-hinge, AOMD-OGS-square, AONS-OGS, AVP-OGS, MAVP-OGS.

The datasets are downloaded from: https://archive.ics.uci.edu/ml/index.php
and 
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html.

AOMD-OGS-hinge is obtained by instantiating AOMD-OGS with the Hinge loss function.
AOMD-OGS-square is obtained by instantiating AOMD-OGS with the square loss function.
FOGD-hinge is obtained by instantiating FOGD with the Hinge loss function.
M-FOGD is a multi-class version of FOGD-hinge.
M-Projectron++ is a multi-class version of Projectron++.
