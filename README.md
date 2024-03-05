# DICOM DRIFT MONITORING DOCKER - README #

This application reads two dicom image distributions, a base distribution with the images used to train a model and a production distribution that refers to new data in which the model is want to be used on. Images related to data drift between the distributions will be outputted to the script directory.

### Methodology ###

The script uses a pytorch pretrained model to get the image features of the dicom images and compares the distributions looking at the euclidean and cosine distances between the baseline and production distributions. The pretrained model can be modified in the script.

The feature vectors are also reduced dimensionally to 2D for visualization of the distributions. The script will output .png files with the relevant plots.

### Requirements ###

Two parameters are required that locate the two text files in which the full path of the dicom series directories to study are listed. One with the baseline images, and the other for the production images to compare with the baseline. 

More parameters related to the analysis and methods can be used with the script, check the dicom_drift_monitor.py script for more info about them.

### Commands ###

The docker is already built. In case it was deleted, it can be built with:

```docker build --no-cache -f dicom_drift_monitoring.dockerfile -t dicom-drift-monitoring-docker .```

Example command to run the docker image interactively:

```docker run -it --rm --user=$USER_ID -v "$DATA_PATH":/data/ -v "$CODE_PATH":/app/ dicom-drift-monitor-docker```

Once inside the docker the script can be run with:

```python dicom_drift_monitor.py --f_baseline $BASE_DICOM_LIST_FILE --f_production $PRODUCTION_DICOM_LIST_FILE```


By: David Vallmanya Poch

Contact: david_vallmanya@iislafe.es
