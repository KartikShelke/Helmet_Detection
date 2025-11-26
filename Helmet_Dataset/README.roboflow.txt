
Bike Helmet Detection - v2 more-preprocessing-augmentation
==============================

This dataset was exported via roboflow.ai on February 18, 2022 at 5:57 PM GMT

It includes 3735 images.
Rider-helmet-bike are annotated in Tensorflow Object Detection format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Random rotation of between -30 and +30 degrees
* Random shear of between -15° to +15° horizontally and -15° to +15° vertically
* Random Gaussian blur of between 0 and 1.5 pixels


