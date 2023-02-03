# MMPD_rPPG_dataset
Here is Mobile Muti-domain Physiological Dataset collected by Tsinghua University.  
The dataset and codes will be uploaded soon with paper publication.

## Examples
<img src="https://github.com/McJackTang/Markdown_images/blob/main/dataset_sample.png?raw=true" width=600 height=800 />

## The Dataset Structure
```
MMPD_videos.tar.gz[wait for edited]
  pxx_x.mat
  ├── video        # Rendered images of the subjects at 320 x 240 resolution     [t, w, h, c]
  ├── GT_ppg       # PPG wavefrom signal                                         [t]
  ├── light        
  ├── motion
  ├── exercise
  ├── skin_color
  ├── gender
  ├── glasser
  ├── hair_cover
  ├── makeup
```
 
Reading the data example:
 
```
import scipy.io as sio
f = sio.loadmat('p1_0.mat')
print(f.keys())
```

## Results
<img src='https://github.com/McJackTang/Markdown_images/blob/main/result1.png' />

## Access and Usage
This dataset is built for academic use. Any commerical usage is banned.  
To access the dataset, you are supposed download this letter of commitment: [wait for upload]. Send an email to <tjk19@mails.tsinghua.edu> with the signed or sealed protocol as attachment.  
There are four kinds of dataset for convenience: full dataset, simple dataset, semi-hard dataset, hard dataset for different kinds of use.  
There are two ways for downloads： OneDrive and Baidu Netdisk for researchers of different regions.  
For those researchers at China, hard disk could also be a solution.

## Reference
The codes are based on this reposity: <https://github.com/ubicomplab/rPPG-Toolbox>
