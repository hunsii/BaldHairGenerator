# BaldHairGenerator
![teaser](overview.jpg)



## Installation
- Clone the repository:
``` 
git clone https://github.com/hunsii/BaldHairGenerator
cd BaldHairGenerator
```
- Dependencies:  


## Download II2S images
Please download the [II2S](https://drive.google.com/drive/folders/15jsR9yy_pfDHiS9aE3HcYDgwtBbAneId?usp=sharing) 
and put them in the `input/face` folder.


## Getting Started  
Preprocess your own images. Please put the raw images in the `unprocessed` folder.
Processed images is in `input/face/` folder.
```
python align_face.py
```


Produce realistic results:
```
python main.py --im_path1 14.png
```

## Acknowledgments
This code borrows heavily from [Barbershop](https://github.com/ZPdesu/Barbersho).
