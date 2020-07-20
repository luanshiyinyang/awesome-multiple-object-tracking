# Awesome Multiple object Tracking: [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

> A curated list of multi-object-tracking and related area resources. It only contains online methods.

<a id="markdown-contents" name="contents"></a>
## Contents

<!-- TOC -->

  - [Review papers](#review-papers)
  - [Algorithm papers](#algorithm-papers)
    - [**2020**](#2020)
    - [**2019**](#2019)
    - [**2018**](#2018)
    - [**2017**](#2017)
    - [**2016**](#2016)
  - [Datasets](#datasets)
    - [Surveillance Scenarios](#surveillance-scenarios)
    - [Driving Scenarios](#driving-scenarios)
  - [Metrics](#metrics)
  - [Benchmark Results](#benchmark-results)
    - [MOT16](#mot16)

<!-- /TOC -->


<a id="markdown-review-papers" name="review-papers"></a>
## Review papers

Multiple Object Tracking: A Literature Review [[paper]](https://arxiv.org/pdf/1409.7618.pdf)

Deep Learning in Video Multi-Object Tracking: A Survey [[paper]](https://arxiv.org/pdf/1907.12740.pdf)

Tracking the Trackers: An Analysis of the State of the Art in Multiple Object Tracking [[paper]](https://arxiv.org/pdf/1704.02781.pdf)


<a id="markdown-algorithm-papers" name="algorithm-papers"></a>
## Algorithm papers


<a id="markdown-2020" name="2020"></a>
### **2020**

**MPNTracker|GMOT**: Learning a Neural Solver for Multiple Object Tracking [[code]](https://github.com/dvl-tum/mot_neural_solver)[[paper]](https://arxiv.org/pdf/1912.07515.pdf)

**UMA**: A Unified Object Motion and Affinity Model for Online Multi-Object Tracking [[code]](https://github.com/yinjunbo/UMA-MOT)[[paper]](https://arxiv.org/pdf/2003.11291.pdf)

**RetinaTrack**: Online Single Stage Joint Detection and Tracking [[code]][[paper]](https://arxiv.org/pdf/2003.13870.pdf)

**FairMOT**: A Simple Baseline for Multi-Object Tracking [[code]](https://github.com/ifzhang/FairMOT)[[paper]](https://arxiv.org/pdf/2004.01888.pdf)

**CenterTrack**: Tracking Objects as Points [[code]](https://github.com/xingyizhou/CenterTrack)[[paper]](https://arxiv.org/pdf/2004.01177.pdf)

**PointTrack**: Segment as points for efficient online multi-object tracking and segmentation [[code]](https://github.com/detectRecog/PointTrack)[[paper]](https://arxiv.org/pdf/2007.01550.pdf)

**PointTrack++**: PointTrack++ for Effective Online Multi-Object Tracking and Segmentation [[code]](https://github.com/detectRecog/PointTrack)[[paper]](https://arxiv.org/pdf/2007.01549.pdf)

**FFT**: Multiple Object Tracking by Flowing and Fusing [[paper]](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2001.11180)

**MIFT**: Refinements in Motion and Appearance for Online Multi-Object Tracking [[code]](https://github.com/nightmaredimple/libmot)[[paper]](https://arxiv.org/pdf/2003.07177.pdf)

**EDA_GNN**: Graph Neural Based End-to-end Data Association Framework for Online Multiple-Object Tracking [[code]](https://github.com/peizhaoli05/EDA_GNN)[[paper]](https://arxiv.org/pdf/1907.05315.pdf)

**GNMOT**: Graph Networks for Multiple Object Tracking [[code]](https://github.com/yinizhizhu/GNMOT)[[paper]](https://link.zhihu.com/?target=https%3A//ieeexplore.ieee.org/stamp/stamp.jsp%3Ftp%3D%26arnumber%3D9093347)


<a id="markdown-2019" name="2019"></a>
### **2019**

**Tracktor++**: Tracking without bells and whistles [[code]](https://github.com/phil-bergmann/tracking_wo_bnw)[[paper]](https://arxiv.org/pdf/1903.05625.pdf)

**DeepMOT**: How To Train Your Deep Multi-Object Tracker [[code]](https://github.com/yihongXU/deepMOT)[[paper]](https://arxiv.org/pdf/1906.06618.pdf)

**JDE**: Towards Real-Time Multi-Object Tracking [[code]](https://github.com/Zhongdao/Towards-Realtime-MOT)[[paper]](https://arxiv.org/pdf/1909.12605v1.pdf)

**MOTS**: MOTS: Multi-Object Tracking and Segmentation[[paper]](https://arxiv.org/pdf/1902.03604.pdf)

**FANTrack**: FANTrack: 3D Multi-Object Tracking with Feature Association Network [[code]](https://git.uwaterloo.ca/wise-lab/fantrack)[[paper]](https://arxiv.org/pdf/1905.02843.pdf)


<a id="markdown-2018" name="2018"></a>
### **2018**

**DeepCC**: Features for Multi-Target Multi-Camera Tracking and Re-Identification [[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Ristani_Features_for_Multi-Target_CVPR_2018_paper.pdf)

**SADF**: Online Multi-Object Tracking with Historical Appearance Matching and Scene Adaptive Detection Filtering [[paper]](https://arxiv.org/pdf/1805.10916.pdf)

**DAN**: Deep Affinity Network for Multiple Object Tracking [[code]](https://link.zhihu.com/?target=https%3A//github.com/shijieS/SST.git)[[paper]](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/1810.11780)

**DMAN**: Online Multi-Object Tracking with Dual Matching Attention Networks [[code]](https://github.com/jizhu1023/DMAN_MOT)[[paper]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ji_Zhu_Online_Multi-Object_Tracking_ECCV_2018_paper.pdf)

**MOTBeyondPixels**: Beyond Pixels: Leveraging Geometry and Shape Cues for Online Multi-Object Tracking [[code]](https://github.com/JunaidCS032/MOTBeyondPixels)[[paper]](http://arxiv.org/abs/1802.09298)

**MOTDT**: Real-time Multiple People Tracking with Deeply Learned Candidate Selection and Person Re-Identification [[code]](https://github.com/longcw/MOTDT)[[paper]](https://arxiv.org/abs/1809.04427)

**DetTA**: Detection-Tracking for Efficient Person Analysis: The DetTA Pipeline [[code]](https://github.com/sbreuers/detta)[[paper]](https://arxiv.org/abs/1804.10134)

**V-IOU**: Extending IOU Based Multi-Object Tracking by Visual Information [[code]](https://github.com/bochinski/iou-tracker/)[[paper]](http://elvera.nue.tu-berlin.de/files/1547Bochinski2018.pdf)


<a id="markdown-2017" name="2017"></a>
### **2017**

**DeepSORT**: Simple Online and Realtime Tracking with a Deep Association Metric [[code]](https://github.com/nwojke/deep_sort)[[paper]](https://arxiv.org/pdf/1703.07402.pdf)

**NMGC-MOT**: Non-Markovian Globally Consistent Multi-Object Tracking [[code]](https://github.com/maksay/ptrack_cpp)[[paper]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Maksai_Non-Markovian_Globally_Consistent_ICCV_2017_paper.pdf)

**IOUTracker**: High-Speed tracking-by-detection without using image information [[code]](https://github.com/bochinski/iou-tracker/)[[paper]](http://elvera.nue.tu-berlin.de/typo3/files/1517Bochinski2017.pdf)

**RNN_LSTM**: Online Multi-Target Tracking Using Recurrent Neural Networks [[code]](https://bitbucket.org/amilan/rnntracking)[[paper]](https://arxiv.org/abs/1604.03635)

**D2T**: Detect to Track and Track to Detect [[code]](https://github.com/feichtenhofer/Detect-Track)[[paper]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Feichtenhofer_Detect_to_Track_ICCV_2017_paper.pdf)

**RCMSS**: Online multi-object tracking via robust collaborative model and sample selection [[paper]](https://faculty.ucmerced.edu/mhyang/papers/cviu16_MOT.pdf)

**towards-reid-tracking**: Towards a Principled Integration of Multi-Camera Re-Identification and Tracking through Optimal Bayes Filters [[code]](https://github.com/VisualComputingInstitute/towards-reid-tracking)[[paper]](https://arxiv.org/pdf/1705.04608.pdf)

**CIWT**: Combined image-and world-space tracking in traffic scenes [[code]](https://github.com/aljosaosep/ciwt)[[paper]](https://arxiv.org/pdf/1809.07357.pdf)


<a id="markdown-2016" name="2016"></a>
### **2016**

**SORT**: Simple online and realtime tracking [[code]](https://link.zhihu.com/?target=https%3A//github.com/abewley/sort)[[paper]](https://arxiv.org/pdf/1602.00763.pdf)


<a id="markdown-datasets" name="datasets"></a>
## Datasets


<a id="markdown-surveillance-scenarios" name="surveillance-scenarios"></a>
### Surveillance Scenarios

PETS 2009 Benchmark Data [[url]](http://www.cvg.reading.ac.uk/PETS2009/a.html)<br>
MOT Challenge [[url]](https://motchallenge.net/)<br>
UA-DETRAC [[url]](http://detrac-db.rit.albany.edu/download)<br>
WILDTRACK [[url]](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/)<br>
NVIDIA AI CITY Challenge [[url]](https://www.aicitychallenge.org/2020-data-and-evaluation/)<br>
VisDrone [[url]](https://github.com/VisDrone)<br>
JTA Dataset [[url]](https://github.com/fabbrimatteo/JTA-Dataset)<br>
Path Track [[url]](https://www.trace.ethz.ch/publications/2017/pathtrack/index.html)<br>
TAO [[url]](https://github.com/TAO-Dataset/tao)<br>


<a id="markdown-driving-scenarios" name="driving-scenarios"></a>
### Driving Scenarios

KITTI-Tracking [[url]](http://www.cvlibs.net/datasets/kitti/eval_tracking.php)<br>
APOLLOSCAPE [[url]](http://apolloscape.auto/tracking.html)<br>


<a id="markdown-metrics" name="metrics"></a>
## Metrics

$$ Accuracy = {{TP + TN} \over {TP + TN + FP + FN}} $$
$$ Recall = {TP \over {TP + FN}} = TPR$$
$$ Precision = {TP \over {TP + FP}} $$
$$ MA = {FN \over {TP + FN}} $$
$$ FA = {FP \over {TP + FP}} $$
$$MOTA = 1 - {\sum_t(FN + FP + IDs)\over \sum_t gt}$$
$$ MOTP = {\sum_{t,i}d_t^i \over \sum_tc_t }$$
$$ IDP = {IDTP \over {IDTP + IDFP}} $$
$$ IDR = {IDTP \over {IDTP + IDFN}} $$
$$ IDF1 = {2 \over {{1 \over IDP} + {1 \over IDR}}} = {2IDTP \over {2IDTP + IDFP + IDFN}} $$

[Evaluation code](https://github.com/cheind/py-motmetrics)

<a id="markdown-benchmark-results" name="benchmark-results"></a>
## Benchmark Results

<a id="markdown-mot16" name="mot16"></a>
### MOT16

| Rank |      Model       | MOTA |                                                    Paper                                                    | Year |
| :--: | :--------------: | :--: | :---------------------------------------------------------------------------------------------------------: | :--: |
|  1   |     FairMOT      | 68.7 |                                 A Simple Baseline for Multi-Object Tracking                                 | 2020 |
|  2   |       JDE        | 64.4 |                                   Towards Real-Time Multi-Object Tracking                                   | 2019 |
|  3   |      Lif_T       | 61.3 |                     Lifted Disjoint Paths with Application in Multiple Object Tracking                      | 2020 |
|  4   |     MPNTrack     | 58.6 |                            Learning a Neural Solver for Multiple Object Tracking                            | 2020 |
|  5   | DeepMOT-Tracktor | 54.8 |                                 How To Train Your Deep Multi-Object Tracker                                 | 2019 |
|  6   |       TNT        | 49.2 |                      Exploit the Connectivity: Multi-Object Tracking with TrackletNet                       | 2018 |
|  7   |       GCRA       | 48.2 | Trajectory Factory: Tracklet Cleaving and Re-connection by Deep Siamese Bi-GRU for Multiple Object Tracking | 2018 |
|  8   |       FWT        | 47.8 |                      Fusion of Head and Full-Body Detectors for Multi-Object Tracking                       | 2017 |
|  9   |      MOTDT       | 47.6 |   Real-time Multiple People Tracking with Deeply Learned Candidate Selection and Person Re-Identification   | 2018 |
|  10  |       NOMT       | 46.4 |                   Near-Online Multi-target Tracking with Aggregated Local Flow Descriptor                   | 2015 |
|  11  |      DMMOT       | 46.1 |                     Online Multi-Object Tracking with Dual Matching Attention Networks                      | 2019 |

### MOT17

| Rank |       Model       | MOTA |                                                    Paper                                                     | Year |
| :--: | :--------------: | :--: | :---------------------------------------------------------------------------------------------------------: | :--: |
|  1   |     FairMOT      | 67.5 |                                 A Simple Baseline for Multi-Object Tracking                                 | 2020 |
|  2   |       Lif_T        | 60.5 |                                   Lifted Disjoint Paths with Application in Multiple Object Tracking                                   | 2020 |
|3|MPNTrack| 58.8 | Learning a Neural Solver for Multiple Object Tracking | 2020|
|4| DeepMOT | 53.7|How To Train Your Deep Multi-Object Tracker|2019|
|5| JBNOT|52.6| Multiple People Tracking using Body and Joint Detections|2019|
|6|TNT|51.9|Exploit the Connectivity: Multi-Object Tracking with TrackletNet|2018|
|7|	FWT|51.3|Fusion of Head and Full-Body Detectors for Multi-Object Tracking|2017|
|8|MOTDT17|50.9|Real-time Multiple People Tracking with Deeply Learned Candidate Selection and Person Re-Identification|2018|

### MOT20


| Rank |       Model       | MOTA |                                                    Paper                                                     | Year |
| :--: | :--------------: | :--: | :---------------------------------------------------------------------------------------------------------: | :--: |
|  1   |     FairMOT      | 61.8 |                                 A Simple Baseline for Multi-Object Tracking                                 | 2020 |
|2| UnsupTrack| 53.6 |Simple Unsupervised Multi-Object Tracking|2020|
