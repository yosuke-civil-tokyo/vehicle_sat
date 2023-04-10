## Folder to contain satellite images  
### Folder Structure(input)  
```
.
├── [place]
│   └── satellite
│       ├── [place]_0
│       │   ├── raw_img
│       │   │   ├── ~~~_PAN (WV2/WV3 image folder)
│       │   │   └── ~~~_MUL (WV2/WV3 image folder)
│       │   ├── mask1
│       │   │   ├── pix_to_edge.csv
│       │   │   ├── mask.tif
│       │   │   ├── mask_qua.tif
│       │   │   ├── mask_f.tif
│       │   │   └── mask_f_qua.tif
│       │   ├── output1
│       │   │   ├── edge_for_sim.csv
│       │   │   └── node_for_sim.csv
│       │   └── preprocessed
│       │       └── panmul
│       │           ├── [place]_PAN.tif
│       │           ├── [place]_NIR.tif
│       │           ├── [place]_MULPAN.tif
│       │           └── [place]_NIR2.tif
│       └── [place]_1
│           ├── raw_img
│           ├── mask1
│           ├── output1
│           └── preprocessed
│               └── panmul
├── [place]
├── [place] 
```  
- check data in mask1 & output1 are output from [road network repository](https://github.com/yosuke-civil-tokyo/cresi_binN)
- for pre-processed satellite imagery, check [sat_process](../sat_process) directory
