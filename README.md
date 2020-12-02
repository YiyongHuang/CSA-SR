# CSA-SR
video captioning based on channel soft attention and semantic reconstructor
### Requirements

* Python 3.6
* TensorFlow 2.0
* Pytorch 1.0
* skimage
* v2
* numpy

###  Architechture

![image](https://github.com/YiyongHuang/CSA-SR/blob/master/Figure1.png)

### Usage

Put videos in `dataset` folder, then extract the video features and linear features with follow commands.

```python
python extract_feats.py
```

```python
python extract_feats_linear.py
```

Training

```shell
python train.py
```

Evaluation

```shell
python evaluate.py
```

The generated results can be evaluated by using the metrics of WangLei(https://github.com/wangleihitcs/CaptionMetrics)

