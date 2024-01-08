## 图像上色 + 图像风格化配置教程
```
1. 图像上色采用的库为 torch和gradio，torch使用的1.1x版本

2. 环境配置：(建议先根据自己的cuda版本 将pytorch安装好，这里采用的是1.13.0+cu117)
cd DDColor
pip install -r requirements.txt
python setup.py develop
cd ..
cd AesUST
pip install -r requirements.txt

3. 下载图像上色的权重文件： modelscope.zip ,解压后得到modelscope文件夹直接放入DDColor文件夹下即可。
权重文件下载链接：https://pan.baidu.com/s/1XKXKpbH3EDDFUxudqXog_g  提取码：1234

4. 下载图像风格化的权重文件：
从 [google drive](https://drive.google.com/file/d/1Ldpfkt32r--ZWwhHSaKbJb7YJbOuwYLZ/view?usp=sharing) 下载， 解压并将它们放入到路径 `AesUST/models/`. 
```