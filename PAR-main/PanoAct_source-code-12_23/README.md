
# European Conference on Computer Vision 2022

The source codes for ECCV2022 Paper: 
Panoramic Human Activity Recognition. 
[[paper]](x)
[[supplemental material]](x)

If you find our work or the codebase inspiring and useful to your research, please cite
```bibtex
x
```
        


## Dependencies
- Python `3.6`
- PyTorch `1.9.0`, Torchvision `0.10.0`
- [RoIAlign for Pytorch](https://github.com/longcw/RoIAlign.pytorch)
- [cdp]()
- [spectral_cluster]()


## Prepare Datasets

1. Download publicly available JRDB dataset from following links: [JRDB dataset](http://vml.cs.sfu.ca/wp-content/uploads/volleyballdataset/volleyball.zip).
2. Save the images under ./PanoAct_source-code/data/imgs
3. Stage1 model weights file and annotation files should be put into ./PanoAct_source-code/data
   We uploaded the base model of stage I to the cloud storage. Put this file into the path ./data. https://pan.baidu.com/s/1eW9uj7wO8vaFgWSoRD-UeA @ PHAR.

## Get Started

1. **Train**: 
    1. **Stage1**
        Set backbone weights file path in config.json, default is: data/stage_1.pth
        ```
        Datasets[JRDB][stage1_model_path]: YOUR_PATH
        ```
       
    2. **Stage2**
        ```
        python train_stage2.py
        ```

2. **Inference**:

    During the second stage of training, the inference results on testing set will be saved as pth files in the `.\result\xxx\xxxepochxxx.pth` folder. Suppose the path of the pth file is `YOUR_RESULT_PATH`, run the following code to get metric scores.
       ```
       python metrics.py YOUR_RESULT_PATH
       ```
