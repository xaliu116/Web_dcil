## 配置教程
```
1. 该应用采用的库为 torch和gradio，torch使用的1.1x版本

2. 环境配置：(建议先根据自己的cuda版本 将pytorch安装好，这里采用的是1.13.0+cu117)
cd DDColor
pip install -r requirements.txt
python setup.py develop
cd ../AesUST
pip install -r requirements.txt
cd ../RapidLaTeXOCR
pip install -r requirements.txt
cd ../first_order_motion
pip install -r requirements.txt

3. 下载权重文件： 将model文件夹内的权重文件放入对应的文件夹中。
权重文件下载链接：https://pan.baidu.com/s/1WbjJofTDm_1YhnNHS6i02w?pwd=dcil  提取码：dcil
```