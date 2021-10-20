# meoknow-ml
## 模型使用(默认以后配置好环境)
*需手动修改文件内参数

```cmd
cfg.MODEL.WEIGHTS="权重文件所在路径"
image =cv2.imread("测试图片所在路径")
```
*运行
```cmd
python3 detect.py
```
*输出

输出图片内猫的预测类别及相应分数
产生after.jpg为预测bounding-box可视化

