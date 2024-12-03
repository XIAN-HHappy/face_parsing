# face parsing
人脸区域分割

## 项目介绍    
注意：该项目不包括人脸检测部分，人脸检测项目地址：https://github.com/XIAN-HHappy/yolo_v3

* 图片示例：  
![image](samples/t.jpg)    

* 视频示例：  
![video](samples/sample.gif)    

## 项目配置  
* 作者开发环境：  
* Python 3.7  
* PyTorch >= 1.5.1  

## 数据集  
* CelebAMask-HQ dataset，数据下载地址：  
  https://github.com/switchablenorms/CelebAMask-HQ

```
• The CelebAMask-HQ dataset is available for non-commercial research purposes only.
• You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data.
• You agree not to further copy, publish or distribute any portion of the CelebAMask-HQ dataset. Except, for internal use at a single site within the same organization it is allowed to make copies of the dataset.

```

* 数据集制作  
  下载数据集并解压，然后运行脚本 prepropess_data.py,生成训练用的mask，注意脚本内相关参数配置。  

## 预训练模型   
  提供512,256两种分辨率的预训练模型。    
* [预训练模型下载地址(百度网盘 Password: ri6m )](https://pan.baidu.com/s/1I5fPAyXDfIh9M5POs80ovg)     

## 项目使用方法   

### 步骤1：生成训练数据
* 目前建议输出2种样本分辨率 512 和 256 （提供512,256两种分辨率的预训练模型），分辨率太小返回原图尺寸时会出现掩码锯齿状，需要后处理解决。    
  注意训练、推理脚本也要做相应的分辨率对应设置。    
* 运行脚本：prepropess_data.py (注意脚本内相关参数配置 )   

### 步骤2：模型训练     
* 根目录下运行命令： python train.py     (注意脚本内相关参数配置 )   

### 步骤3：模型推理    
* 根目录下运行命令： python inference.py   (注意脚本内相关参数配置  )  
