## 图像上色
```
1. 图像上色采用的库为 torch和gradio，torch使用的1.1x版本
2. 环境配置：

pip install -r requirements.txt
python setup.py develop

3. 下载权重文件： modelscope.zip ,解压后得到modelscope文件夹直接放入DDColor文件夹下即可。
权重文件下载链接：https://pan.baidu.com/s/1XKXKpbH3EDDFUxudqXog_g  提取码：1234
```

```
文件存在两种情形：

1. 上传单张图像，此需要单张展示

2. 上传files(当然也是可以只有一张图像), 返回一个文件夹或者zip
```