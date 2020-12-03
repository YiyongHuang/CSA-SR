# CSA-SR
video captioning based on channel soft attention and semantic reconstructor
### Requirements
* Python 3.6
* TensorFlow 2.0
* Pytorch 1.0
* skimage
* openCV
* numpy

###  Architechture

![image](https://github.com/YiyongHuang/CSA-SR/blob/main/Architechture.jpg)

### Usage
Put videos in `dataset` folder and alter the path to video in `extract_feats.py` and `extract_feats_linear.py`, then extract the video features and linear features with follow commands.
```python
python extract_feats.py
```
```python
python extract_feats_linear.py
```

##### Training
Alter the imported package `ConvGRU` with `ConvGRU_att` in `CSA-SR.py` can change the module.
```shell
python train.py
```

##### Evaluation
Alter the imported package `CSA_SR_beam` with `CSA_SR` in `evaluate.py` can evaluate model using beam search.
```shell
python evaluate.py
```

The generated results can be evaluated by using the metrics of WangLei(https://github.com/wangleihitcs/CaptionMetrics)

