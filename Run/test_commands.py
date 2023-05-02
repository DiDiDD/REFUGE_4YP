python3 Model_test.py {test_num} 5e-5 15 1 'unet'  'batch'  --base_c 12

python3 Model_test.py {test_num} 5e-5 15 0 'unet'  'batch'  --base_c 24

python3 Model_test.py {test_num} 5e-5 15 1 'unet'  'batch'  --base_c 32

python3 Model_test.py {test_num} 5e-5 15 0 'unet'  'batch'  --base_c 48

python3 Model_test.py {test_num} 5e-4 15 1 'swin_unetr' 'layer'  --base_c 24 --depth '[1,1,2,2]'

python3 Model_test.py {test_num} 5e-5 15 0 'swin_unetr' 'instance'  --base_c 24 --depth '[1,1,2,2]'

python3 Model_test.py {test_num} 1e-4 15 1 'utnet'  'batch'  --base_c 3

python3 Model_test.py {test_num} 1e-4 15 0 'utnet'  'batch'  --base_c 6

python3 Model_test.py {test_num} 1e-4 15 1 'utnet'  'batch'  --base_c 12

python3 Model_test.py {test_num} 1e-4 15 0 'utnet'  'batch'  --base_c 18

python3 Model_test.py {test_num} 1e-4 20 1 'utnet'  'batch'  --base_c 12

python3 Model_test.py {test_num} 1e-4 15 0 'utnet'  'instance'  --base_c 6

python3 Model_test.py {test_num} 1e-4 15 1 'utnet'  'instance'  --base_c 12

python3 Model_test.py {test_num} 1e-4 15 0 'utnet'  'layer'  --base_c 6

python3 Model_test.py {test_num} 1e-4 15 1 'utnet'  'layer'  --base_c 12

python3 Model_test.py {test_num} 5e-5 15 0 'swin_unetr' 'instance'  --base_c 24 --depth '[1,1,2,2]'

python3 Model_train.py {test_num} 1e-5 15 1 'swin_unetr' 'batch'  --base_c 12 --depth '[1,1,2,2]'

python3 Model_train.py {test_num} 1e-5 15 0 'swin_unetr' 'layer'  --base_c 12 --depth '[1,1,2,2]'
