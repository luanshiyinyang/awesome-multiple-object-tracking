# 多目标跟踪
<!-- TOC -->


- 论文
- 数据集
  - PETS 2009 Benchmark Data
  - MOT Challenge
- 指标
- 基准结果

<!-- /TOC -->
## 论文

## 数据集

以下是监视视角下的数据集
### PETS 2009 Benchmark Data

该数据集是一个较老的数据集，发布与2009年，是包含不同人群活动的多传感器序列，可以用于估计人群人数和密度，跟踪人群中的个人以及检测流量和人群事件。
数据集具体结构如下：

+ 校正数据
+ S0：训练数据
  + 包含设置背景，市中心，常规流量
+ S1：人数和人群密度估计
  + 包含：L1,L2,L3
+ S2：人物跟踪
  + 包含：L1,L2,L3
+ S3：流分析和事件识别
  + 包含：事件识别和多重流

可用于多目标跟踪的是S2部分，从L1到L3，人群密度逐渐增大，困难程度变大。但在处理多个视图的时候，需要用到相机校正数据，将每个人的2D边界框投影到其他视图中。

### MOT Challenge

MOT Challenge是多目标跟踪方向一个很有影响力的比赛，专注于行人跟踪。其从2015年开始提供用于行人跟踪的数据集，至今包含2D MOT 2015、MOT16、MOT17、MOT20、MOTs。还有用于检测的MOT17Det和MOT20Det，以及用于石斑鱼跟踪的3D-ZeF20。
#### MOT20
用最新的MOT20举例，MOT20包含4组训练用的序列以及4组测试用的序列。
![MOT20训练集](/assets/MOT20_trainset.jpg)
![MOT20训练集](/assets/MOT20_testset.jpg)
MOT的标签文件分为用于检测的标签和ground truth两种，均为txt格式存储。首先是用于检测的标签，其标注格式为：
<br>`<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>`
<br>例如：<br>
1. 1,-1,757,692,96,209,1,-1,-1,-1
2. 1,-1,667,682,100,222,1,-1,-1,-1
3. 1,-1,343,818,127,258,1,-1,-1,-1

第一个数字是代表帧数；第二个数字-1，意味着没有分配ID；随后的两个数字分别是Bbox的左上角点的坐标；再接着的两个数字是Bbox的w和h；后一个数字表示的是置信度；最后三个-1对检测文件来说没有意义。<br>ground truth的标注格式为：
<br>`<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <0/1>, <cls>, <vis>`
<br>例如：<br>
1. 1,1,199,813,140,268,1,1,0.83643
2. 2,1,201,812,140,268,1,1,0.84015
3. 3,1,203,812,140,268,1,1,0.84015

第一个数字依旧代表着帧数；第二个数字是该Bbox的ID；后面四个数字是Bbox的位置以及大小信息，同上；后一个数字表示的也是置信度，0代表着ignored，1代表着considered；再后一个数字代表着类别；最后一个数字代表着该目标的可视度（遮挡或者处于图像边界会造成目标部分不可见），值的范围是0~1，
#### MOTs
MOTs是德国亚琛工业大学计算机视觉实验室在2019年发布的提出多目标跟踪与分割的网络TrackR-CNN的文章时一同发布的数据集。MOTs数据集是基于KITTI_Tracking和MOT_Challenge重新标注的多目标跟踪与分割数据集，是像素级别的数据集。目前只有行人和车辆两个分类。其GitHub地址为[mots_tools](https://github.com/VisualComputingInstitute/mots_tools)。
<br>MOTs数据集提供了png和txt两种编码格式。两种格式中id值为10000都表示着忽略区域。
<br>png格式
<br>png格式具有16位的单颜色通道，可通过以下代码读取：
```
import PIL.Image as Image
img = np.array(Image.open("000005.png"))
obj_ids = np.unique(img)
% to correctly interpret the id of a single object
obj_id = obj_ids[0]
class_id = obj_id // 1000
obj_instance_id = obj_id % 1000
```
或者采用TensorFlow时，可以采用如下代码：
```
ann_data = tf.read_file(ann_filename)
ann = tf.image.decode_image(ann_data, dtype=tf.uint16, channels=1)
```
txt格式
txt文件中的格式为time_frame，id，class_id，img_height，img_width，rle，rle为COCO中的编码。<br>例如：<br>1 2029 2 1080 1920 kWn[19ZQ1;I0C>000000000000O13M5K2N00001O001O00001O1O005Df\`b0
<br>这代表着第1帧，目标id为2029（分类id为2，即行人；实例id为29），图片大小为1080*1920。这种格式的文件也可以采用[cocotools](https://github.com/cocodataset/cocoapi)进行解码。

### UA-DETRAC



## 指标

## 基准结果