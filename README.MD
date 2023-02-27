# Transformer Research for Medical Image Segmentation

### Command line:
python3 Model_train.py {number: learning rate} {int: batch size} {int: GPU index} {str: 'model name'} {str: 'model description'}\
eg: '''console python3 Model_train.py 5e-5 15 1 'unet'  'unet_instNorm' '''

python3 Model_test.py {number: learning rate} {int: batch size} {int: GPU index} {str: 'model name'} {str: 'model description'}\
Important: The 'model description' must be one of those you specifying for training.\
eg: python3 Model_test.py 5e-5 15 1 'unet'  'unet_instNorm'

