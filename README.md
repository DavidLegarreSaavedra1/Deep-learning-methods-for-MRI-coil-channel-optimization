# Deep learning methods for MRI coil channel optimization

# Introduction

This repository contains the code developed by David Legarre Saavedra for his Bachelor's degree final project.
This Final project was done in collaboration with General Electric and L'Hospital Clínic de Barcelona.

Magnetic resonance imaging (MRI) scanners use powerful magnets and make use of the
human body reactions to such fields to produce images. Modern MRI uses multi-channel
receiver coils to enable accelerated data acquisition. With the introduction of parallel
image acquisition techniques, the time it takes for a scan of this type has been significantly
reduced. Despite their many advantages, acceleration techniques cause a number of image
artifacts, particularly in the presence of patient motion. In particular, we will aim at
improving the diagnostic quality of clinical 3D myocardial delayed enhancement (MDE)
data by optimizing the set of coil channels used in the reconstruction of the images. We
will implement and use the ROVir algorithm to remove these artifacts from the image,
and implement a DL approach to automatize its parameters.

# How to use

* If it is the first time you run this project. 

    `pip install -r requirements.txt`

* Then add the set of slices to test the algorithm in *data/* repository. The slices should in the NIFTI format.
* Run `main.py`
