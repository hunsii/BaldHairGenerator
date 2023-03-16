# BaldHairGenerator
![teaser](docs/assets/teaser.png)



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
```
python align_face.py
```

Produce realistic results:
```
python main.py --im_path1 90.png
```


## Todo List
* add a detailed readme
* update mask inpainting code
* integrate image encoder
* add preprocessing step
* ...

## Acknowledgments
This code borrows heavily from [II2S](https://github.com/ZPdesu/II2S).
