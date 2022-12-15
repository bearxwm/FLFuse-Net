# The code of paper 'FLFuse-Net: A Fast and Lightweight Infrared and Visible Image Fusion Network Via Feature Flow and Edge Compensation for Salient Information'.


## Paper: https://doi.org/10.1016/j.infrared.2022.104383
## Network Structure: 
### NET: ![Structure of the proposed FLFuse-Net](./fig/NET.png)
### edge compensation branch: ![Edge Compensation Branch](./fig/S_Branch.png)

## Env
pytorch=1.8.0 python=3.7 

then pip the needed packages

## Run
### Test
1. Edit the test data root in data_loader.py with testpath()
2. Run test.py

### Train
1. Use dataset at http://matthewalunbrown.com/nirscene/nirscene.html.
2. Set run Parameters in main.py
3. Replace the [your_env/site-packages/torchvision/transforms/transforms.py] with [1.py]
4. Run main.py
