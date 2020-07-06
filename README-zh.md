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
UA-DETRAC是一个车辆多目标检测和跟踪的数据集。数据集包含了在中国北京和天津24个不同地点使用Cannon EOS 550D摄像机拍摄的10个小时的视频。视频以每秒25帧（fps）的速度录制，分辨率为960×540像素。UA-DETRAC数据集中有超过14万个帧，并且有8250辆车进行了手动注释，因此总共有121万个带标签的对象边界框。下载地址为[UA-DETRAC](http://detrac-db.rit.albany.edu/download)。数据集结构如下：
+ 数据集
  + 训练集图像（5.22GB，60个序列）
  + 测试集图像（3.94GB，40个序列）
+ 检测
  + 训练集检测（DPM, ACF, R-CNN, CompACT）
  + 测试集检测（DPM, ACF, R-CNN, CompACT）
+ 注释
  + DETRAC-Train-Annotations-XML：包含带有属性信息（例如，车辆类别，天气和比例）的完整注释，该注释用于检测训练。
  + DETRAC-Train-Annotations-MAT：包含数据集中忽略背景区域之外的目标轨迹的位置信息，用于检测和跟踪评估。
  + DETRAC-Train-Annotations-XML-v3：包含具有属性信息（例如，车辆类别和颜色，天气和比例）的改进注释，该注释用于检测，跟踪和计数训练。
  + DETRAC-Sequence-Locations：包含每个序列的特定位置信息（24个不同的位置）。
  + DETRAC-Test-Annotations-XML：包含具有属性信息（例如，车辆类别，天气和比例）的完整注释，该注释用于检测训练。
  + DETRAC-Test-Annotations-MAT：包含目标轨迹在数据集中忽略背景区域之外的位置信息，用于检测和跟踪评估。 

其中，DETRAC-Train-Annotations-XML文件如下：
![DETRAC-Train-Annotations-XML](assets/DETRAC-Train-Annotations-XML.png)
DETRAC-Train-Annotations-MAT文件是.mat格式存储，只包含了目标的边界框。测试集的格式与训练集相同。
<br><br>UA-DETRAC数据集绘制之后的情况如下：
![UA-DETRAC](assets/UA-DETRAC.png)
红色框表示车辆完全可见，蓝色框表示车辆被其他车辆遮挡，粉色矿表示车辆被背景部分遮挡。左下角为该段序列的天气状况、摄像机状态和车辆密度的信息。
<br>[UA-DETRAC](http://detrac-db.rit.albany.edu/download)还提供了数据集的评估工具，有用于评估多目标检测的，也有用于多目标跟踪的。该工具包采用Matlab编程，可以用来绘制PR曲线。

### WILDTRACK
该数据集采用七个具有重叠视场的高科技静态定位相机获取的，具有高度精确的联合摄像机校准以及视图序列之间的同步。视频的分辨率为1920×1080像素，以每秒60帧的速度拍摄。
<br>数据集中包含：
1. 以10帧/秒，1920×1080分辨率的帧速率提取的同步帧，并经过后处理来消除失真。
2. 相机模型的校准文件，与OpenCV库中提供的投影功能兼容。
3. json文件格式的地面注释。
4. json文件格式的position文件，方便注重于分类的算法使用。

<br>下载地址在[WILDTRACK](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/)。

### NVIDIA AI CITY Challenge
NVIDIA AI CITY Challenge是NVIDIA公司举办人工智能城市挑战赛，分为四场比赛：运动车辆计数、车辆重识别、多目标车辆跟踪和交通异常检测。每个比赛都提供了专用的数据集，其中可以用于车俩多目标跟踪的是City-Scale Multi-Camera Vehicle Tracking。
<br>此数据集大小为15.7个GB，包含215.03分钟的视频，这些视频是从46个摄像机跨越美国中型城市的16个交叉路口收集到的。两个最远的同时摄像头之间的距离为4km。该数据集涵盖了多种位置类型，包括交叉路口，道路延伸和公路。数据集1/2为训练集，1/3为验证集，1/6是测试集。总体而言，数据集包含了近38万个边界框，用于880个不同的带注释的车辆标识，并且仅注释了通过至少2个摄像机的车辆。每个视频的分辨率至少为960p，大多数视频的FPS为10。此外，在每种情况下，每个视频都可以使用从开始时间开始的偏移量来同步。
<br>下载地址为[NVIDIA AI CITY Challenge](https://www.aicitychallenge.org/2020-data-and-evaluation/)


## 指标

## 基准结果