# Transformer Research for Medical Image Segmentation

### GitHub Repository
https://github.com/DiDiDD/REFUGE_4YP.git


### Requirements
Install all the packages required by running: 
```
pip install -r requirements.txt
```


### Data
We used data from REFUGE Challenge (Retinal Fundus Glaucoma Challenge). Download the data from: 

https://refuge.grand-challenge.org/Home2020/

Merge the data in train/val/test if the images/masks are separated into Glaucoma and Non-Glaucoma.


### Data processing
Firstly run:
```
python3 REFUGE_4YP/Run/data_aug/data_aug.py
```
Note: Change the code from line 81 - 87 to match the path to the merged data folder you specified.

The other data augmentation files located in the 'data_aug' folder serve the purpose of investigating the model's robustness.



### To train models:
```python3 Model_train.py  {float: learning rate}  {int: batch size}  {int: GPU index}  {str: 'model name'} {str: Normalisation Name}```\
eg:
```
cd REFUGE _4YP/Run
python3 Model_train.py 5e-5 15 1 'unet' 'batch' 
```


### To test models:
```python3 Model_test.py {int: test data number} {float: learning rate}  {int: batch size}  {int: GPU index}  {str: 'model name'}  {str: Normalisation Name} ```\
eg:
```
cd REFUGE _4YP/Run
python3 Model_test.py 1 5e-5 15 1 'unet' 'batch' 
```


# On Machine
The augmented data is located in Desktop/new_data/

The GitHub repository clone is located in Desktop/REFUGE_4YP/

The results (checkpoints, test scores, prediction masks) are located in Desktop/new_data/test/

### To check results:
Step into one of the folders in Desktop/new_data/test/ and select which test set. \
eg:
```
cd Desktop/new_data/test/1600_unet_batch_lr_5e-05_bs_15_fs_12/results0
```
The prediction mask shall be observed and test scores is in 'npy' file for the test set.

