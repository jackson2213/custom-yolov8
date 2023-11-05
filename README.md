# onnx-android-yolov8

最新YOLOV8 ONNX模型导入安卓，无需转换NCCN



## how to build and run
### step1
打开YOLOV8官方网站: https://github.com/ultralytics/ultralytics

### step2
使用pip安装最新YOLOV8
```
pip install ultralytics
```


### step3
* 训练自己的模型,可参考YOLOV8官方
```
from ultralytics import YOLO
from multiprocessing import freeze_support

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

if __name__ == '__main__':
freeze_support()  # for Windows support

    model.train(data="fire.yaml", epochs=100)  # train the model
```
### step4
* 导出自己的模型,可参考官方
```
from ultralytics import YOLO
model = YOLO('best.pt')  # load a custom trained model
# Export the model
model.export(format='onnx',simplify=True)
``` 
### step5
* 在安卓项目中添加runtime包
* implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.12.1'

### step6
* 把导出的onnx模型放入assets文件夹下，具体调用可看MainActivity load_model函数

## some notes
* yolo v5 模型 无法运行。 请参阅其他 GitHub 站点。


## Reference：  
https://github.com/Aloe-droid/Yolov8_Android


