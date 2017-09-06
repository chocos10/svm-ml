# SVM FOR CLASSIFICATION OF CANCEROUS CELLS

This project demonstrates the implementation of Support Vector Machine (SVM) for the purpose of understanding its  principles and issues for classification

## Data

The data used in this project is the Wisconsin Diagnostic Breast Cancer (WDBC) Data Set1. The samples were obtained by Fine Needle Aspiration (FNA) Biopsy, which is a procedure that involves passing a thin needle through the skin to sample fluid or tissue from a cyst or solid mass, as illustrated in Figure 1. The sample of cellular material taken during an FNA is then sent to a pathology laboratory for analysis. Figure 2 shows an image of cells obtained from FNA.

Fig. 1. Illustration of FNA Biopsy

<img src="images/f1-fna-biopsy.png" alt="Fig. 1. Illustration of FNA Biopsy" width="320" />

Fig. 2. Image of cells from FNA

<img src="images/f2-cells-fna.png" alt="Fig. 2. Image of cells from FNA" width="320" />



Each sample has 30 real-valued features that have been computed from a digitized image of a fine needle aspirate of a breast mass. They describe characteristics of the cell nuclei present in the image. In addition to these 30 features, each sample also contains a number (which serves
as an ID for the sample) and a label of either B (for benign) or M (for malignant). Thus, each type of label defines a class.

As an illustration, two samples are shown below. The first number in each sample is the ID, followed by the label and then by the 30 values for the features.

>883852, B, 11.3, 18.19, 73.93, 389.4, 0.09592, 0.1325, 0.1548, 0.02854, 0.2054, 0.07669, 0.2428, 1.642, 2.369, 16.39, 0.006663, 0.05914, 0.0888, 0.01314 0.01995, 0.008675, 12.58, 27.96, 87.16, 472.9, 0.1347, 0.4848, 0.7436, 0.1218, 0.3308, 0.1297

>842302, M, 17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189

Two group of samples have been selected from the original WDBC data set to form a training set (with 285 samples) and a test set (with 100 samples). Another group of 100 samples are then selected from the remaining samples in the WDBC data set to form an evaluation set. This evaluation set is not available to the students, and is to be used to evaluate the performance of the SVM program submitted by the students, as part of the assessment scheme described later in Section V.

The training set and the test set are included in the zipfile that also contains this document. They are in the MATLAB MAT-file format, with the file names train.mat and test.mat. They can be directly loaded into MATLAB. Each of these two MAT-files contains two variables: the matrix variable data in which each column represents one sample, and the variable label which contains the class label (i.e., +1 for B and -1 for M) of the samples.

## Dataset

The dataset resulting from the samples obtained by a Fine Needle Biopsy have 30 attributes - ID, diagnosis and 30 real-valued input features.

### Attribute information
```
1. ID number
2. Diagnosis (M = malignant, B = benign)
3. Ten real-valued features are computed for each cell nucleus:
  a) radius (mean of distances from center to points on the perimeter)
  b) texture (standard deviation of grey-scale values)
  c) perimeter
  d) area
  e) smoothness (local variation in radius lengths)
  f) compactness (perimeter^2 / area - 1.0)
  g) concavity (severity of concave portions of the contour)
  h) concave points (number of concave portions of the contour)
  i) symmetry
  j) fractal dimension ("coastline approximation" - 1)
The nucleus of a cancerous cell is often larger and darker than of a normal cell and its shape and size is not uniform.
```
## Support Vector Machine

There are many ways to separate data. However, is there an optimal way? The Support Vector Machine theory provides a systematic method for separating data optimally. The focus is on finding the margin which gives a maximum distance between the separable classes.

Compared to MLP, SVM has following advantages –
1. Solution found by SVM is optimal where as MLP finds one of the many solutions.
2. Structure of SVM is simple.
3. The solution process is tractable whereas MLP does not guarantee convergence.

The solution process of Support Vector Machine (SVM) focuses on finding a hyperplane that divides a set of samples into two categories (here, benign and malign cells). To achieve this, we must find a hyperplane which keeps the samples as far away as possible.

The margin can be of two types –
1. **Hard margin**: This margin works best for data set where the samples strictly do not fall inside the region of separation.
2. **Soft margin**: This margin works best for data set where samples may not linearly separable.

Fig 3. Optimal hyperplane with hard margin

<img src="images/f3-op-hm.png" alt="Fig 3. Optimal hyperplane with hard margin" width="320" />

Fig 4. Optimal hyperplane with soft margin

<img src="images/f4-op-sm.png" alt="Fig 4. Optimal hyperplane with soft margin" width="320" />

Another approach for sample that are not linearly separable is to transform them to higher dimensions. This increases the probability of the samples being linearly separable in higher dimensional space. The transformation can be achieved by using a kernel. For this classification, we use linear and polynomial kernels.

Fig 5. Transformation to higher dimensional space using a kernel

<img src="images/f5-fx-using-kernel.png" alt="Fig 5. Transformation to higher dimensional space using a kernel" width="640" />
