## pre-process & post-process
Run them from the 'vehicle' directory (one upper from here)
- preprocessing of raw satellite imagery  
    preprocess the raw satellite imagery and output the image with given range, size, band
    ```
    python sat_process/preprocess_mulpan.py -data [your data folder('data)] -ulx [top left longutude] -uly [top left latitude] -lrx [bottom right longitude] -lry [bottom right longitude] -place [name your region] -disaster [0 or 1] -mul2 [True if you have 8band multispectrum image]
    ```
- create the .geojson format data from output  
    this repository's output & [road disaster repository](https://github.com/yosuke-civil-tokyo/SAR_disaster_Tellus)'s output can be aggregated & converted to .geojson format  
    you can visuallize .geojson & .geotiff on web app or GIS app  
    ```
    python sat_process/make_json.py -data [your data folder('data)] -ulx [top left longutude] -uly [top left latitude] -lrx [bottom right longitude] -lry [bottom right longitude] -place [name your region] -disaster [0 or 1]
    ```
- merge all the tile images into one large .geotiff
    ```
    python sat_process/merge.py -data [your data folder('data)] -ulx [top left longutude] -uly [top left latitude] -lrx [bottom right longitude] -lry [bottom right longitude] -place [name your region] -disaster [0 or 1]
    ```