<h1 align="center" style="color:blue; font-size:32px;">
    <b>Two-View Left Ventricular Segmentation and Ejection Fraction Estimation in 2D Echocardiograms</b>
</h1>

## Author's Note:

Frank Cally Tabuco is the sole developer, researcher, and writer of this work in 2D echocardiogram segmentation and automated ejection fraction estimation. Rest of the authors are panels and advisers.

Official implementation of the paper: "Two-View Left Ventricular Segmentation and Ejection Fraction Estimation in 2D Echocardiograms" submitted at the 33rd British Machine Vision Conference.

## Abstract
<p align="justify">
Early detection of cardiovascular diseases through assessment of cardiac function is vital for accurate diagnosis, timely treatment, and improved prognosis. Among the various methods for estimation of ventricular function, 2D echocardiography remains to be one of the most valuable, accessible, and practical modalities in clinical practice. However, three main problems have persisted in the assessment of left ventricular (LV) ejection systolic function through ejection fraction (EF) measurement. First, current methods for analysis requires a series of procedures which are labor-intensive, time-consuming, and require high-level of skills to perform correctly. Second, semantic segmentation in 2D echocardiography often deals with low-quality, low-contrast images. Last, estimation of EF suffers from high inter-observer variability reaching as high as 14% error. To solve these problems, we developed segmentation and action recognition models in two-view 2D echocardiography for the automatic semantic segmentation of LV regions and estimation of LV EF. The segmentation model named channel-separated and dilated dense-Unet (CDDenseUnet) is capable of predicting segmented frames which outperformed current state-of-the-art architectures in terms of dice score, mean surface distance, and run-time performance reaching scores of 95.2%, 1.2mm, and 0.02 seconds, respectively. On the other hand, the prediction model named Two-Channel R(2+1)d is capable of analyzing segmented LV regions from echocardiogram videos in apical 2-chamber (A2C) and apical 4-chamber (A4C) views which produces better results than traditional estimation of EF reaching a mean absolute error of 3.8%. These new models have the potential to vastly improve LV EF measurement for the diagnosis of a wide variety of cardiac conditions and find great utility especially in complicated clinical scenarios or limited resource-settings where echocardiograms are prone to generation of sub-optimal image quality.
</p>

## Paper link
- Official: https://bmvc2022.mpi-inf.mpg.de/0176.pdf
