import os

import gradio as gr
import tempfile
import shutil


def generate_file(file_obj):
    global tmpdir
    print('临时文件夹地址：{}'.format(tmpdir))
    print('上传文件的地址：{}'.format(file_obj.name)) # 输出上传后的文件在gradio中保存的绝对地址

    #获取到上传后的文件的绝对路径后，其余的操作就和平常一致了

    # 将文件复制到临时目录中
    shutil.copy(file_obj.name, tmpdir)

    # 获取上传Gradio的文件名称
    FileName=os.path.basename(file_obj.name)

    # 获取拷贝在临时目录的新的文件地址
    NewfilePath=os.path.join(tmpdir,FileName)
    print(NewfilePath)

    # 打开复制到新路径后的文件
    with open(NewfilePath, 'rb') as file_obj:

        #在本地电脑打开一个新的文件，并且将上传文件内容写入到新文件
        outputPath=os.path.join(tmpdir,"New"+FileName)
        with open(outputPath,'wb') as w:
            w.write(file_obj.read())

    # 返回新文件的的地址（注意这里）
    return outputPath
def main():
    global tmpdir
    with tempfile.TemporaryDirectory(dir='.') as tmpdir:
        # 定义输入和输出
        inputs = gr.components.File(label="上传文件")
        outputs = gr.components.File(label="下载文件")

        # 创建 Gradio 应用程序g
        app = gr.Interface(fn=generate_file, inputs=inputs, outputs=outputs,   title="文件上传、并生成可下载文件demo",
                      description="上传任何文件都可以，只要大小别超过你电脑的内存即可"
      )

        # 启动应用程序
        app.launch(share=True)
if __name__=="__main__":
    main()

