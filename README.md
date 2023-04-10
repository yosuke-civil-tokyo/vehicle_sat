# Vehicle detection & speed estimation from Satellite Imagery
This repository is based on [yolov3](https://github.com/nekobean/pytorch_yolov3). Detect vehicles from satellite images and estimate their speed.  
Train the Yolov3 model which detect vehicles from satellite/aerial images. Use the model for detection and judge the vehicle is on road or not. For the vehicle on road, calculate speed using multispectrum satellite imagery.  

Speed estimation is based on the concept that multi-spectrum satellite imagery contains a time lag in different bands (check papers like [this](https://www.mdpi.com/2072-4292/6/7/6500)). In our code, simulation/network information corrects the simple correlation-based speed estimation.  

We put several pre-process & post-process for satellite imagery in [sat_process](sat_process/).

## Prerequisites
To obtain the result from this program, all you need is as follows.  
- Satellite imagery (we assume Maxar satellite, World-View2 or World-View3)
    - Multi-spectral image (RGBN)
    - Panchromatic image
    - (not necessary) Multi-spectral image (other 4 multi-spectral)
- (not necessary) network data obtained from another satellite image repository
    - edge & node information
    - road mask information  

Be aware that we use each raw satellite imagery, not pansharpened(pre-processed) image.

## Process
1. Prepare data & Train the model
2. Detect vehicles & Estimate speed

### Prepare data
For detailed data folder structure, check [README in data](data/).  
We provide two ways to prepare training data for vehicle detection from satellite image.  
1. Use [VEDAI](https://downloads.greyc.fr/vedai/) dataset  
VEDAI dataset is an open source dataset for vehicle detection from satellite/aerial image. To use the VEDAI dataset for our training,  
    - first download (512 version) 'annotations'(annotation text file) and **unzipped** 'part2'(image files) from [here](https://downloads.greyc.fr/vedai/) and put them in 'custom_dataset' folder
    - then run below for making training folder  
    ```
    python edit_for_vedai.py
    ```
2. Use your own satellite image as training data  
You can use your annotation application such as [Vott](https://github.com/microsoft/VoTT) to annotate your satellite imagery.  
    - first you clip your .tif large tile into .png 8-bit image of training size.  
    ```
    python tif_to_png.py -img [your satellite image path] -vott [your vott folder path]
    ```
    - and annotate accordingly
    - make training folder
    ```
    python convert_vott_dataset.py [your vott folder path] [your custom_dataset path]
    ```
    - we recommend that you exclude images with no vehicles in them
    ```
    python arrange_custom.py -custom [your custom_dataset path]

### Train the model
Just follow the Yolov3 training document. See [original directory](https://github.com/nekobean/pytorch_yolov3).  
First download pre-trained weights  
```
cd weights
wget https://pjreddie.com/media/files/darknet53.conv.74
cd ..
```
and run training
```
python train_custom.py --dataset_dir custom_dataset --weights weights/darknet53.conv.74 --config config/yolov3_custom.yaml
```
If you want to check your trained model, run below for any of your images.  
```
python run_newimage.py [your image file name in 'custom_dataset/image']
```

### Detect vehicles and estimate speed  
Run below.  
```
python main.py -data [your data dir('data')] -ulx [top left longutude] -uly [top left latitude] -lrx [bottom right longitude] -lry [bottom right longitude] -place [name your region] -disaster [0 or 1] -mul2 [True if you have 8band multispectrum image]
```
```-disaster``` flag is for distinguishing images on before or after disaster if you have both. ```-mul2``` flag is for checking if you have 8band multispectrum image or RGBN multispectrum image.

---------------
Now we have end-to-end script for vehicle detection & speed estimation. We assume that we have network information in edge/node format & road mask information (mentioned 'not necessary' in prerequisites), and distribute vehicles on network for utilizing network information in speed vector estimation.  
For the satellite image, the code can be adopted to both case, namely 'when we have RGBN + PAN' and 'when we have 8bands + PAN'.  

A more flexible form of function is to be implemented.

