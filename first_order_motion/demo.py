import argparse
import stage1
import stage2
import gradio as gr
import os
import cv2

def main():
    """
    主函数，从终端调用。
    """
    parser = argparse.ArgumentParser(description='运行 demo.py，输入图像和视频')
    parser.add_argument('--src_img', default="obama.jpg")
    parser.add_argument('--src_vid', default="singing_2.mp4")

    args = parser.parse_args()

    pathIn = '/'

    # 流程阶段1：输入预处理
    # 1. 将输入视频和图像放到 first-order-motion/
    stage1.cropVideo(args.src_vid, pathIn)
    stage1.cropImage(args.src_img)

    # 阶段2：图像动画
    stage2.generate_output()


def run_from_server_old(videoPath, imagePath):
    """
    从服务器调用的函数。
    """
    print("服务器成功连接到模型。")
    print("正在预处理输入。")
    pathIn = '/'
    # 流程阶段1：输入预处理
    # 1. 将输入视频和图像放到 first-order-motion/
    res1 = stage1.cropVideo(videoPath, pathIn)
    res2 = stage1.cropImage(imagePath)

    if res1 and res2:
        print("正在生成输出。")
        # 阶段2：图像动画
        result = stage2.generate_output()
        print("输出生成成功。")

        # 生成的视频的绝对路径
        video_path = os.path.abspath(result)

        return (0, video_path)
    else:
        if not res1 and not res2:
            # 图像和视频均未检测到单个人脸
            return (3, "图像和视频均未检测到单个人脸")
        elif not res1:
            # 视频未检测到单个人脸
            return (1, "视频未检测到单个人脸")
        elif not res2:
            # 图像未检测到单个人脸
            return (2, "图像未检测到单个人脸")


def save_image_and_run(video_path, image_array):
    pathIn = '/'
    # 1. 获取视频文件名（不包含扩展名）作为保存图片的文件夹名
    video_filename = os.path.splitext(os.path.basename(video_path))[0]

    # 2. 创建保存图片的文件夹
    save_folder = f"{video_filename}_images"
    os.makedirs(save_folder, exist_ok=True)

    # 3. 生成保存图片的完整路径
    image_filename = f"{video_filename}_image_{len(os.listdir(save_folder)) + 1}.png"
    save_path = os.path.join(save_folder, image_filename)

    # 4. 保存图片
    cv2.imwrite(save_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))

    print(save_path)
    print(video_path)

    res1 = stage1.cropVideo(video_path, pathIn)
    res2 = stage1.cropImage('./'+save_path)
    result = stage2.generate_output()

    # 5. 返回保存路径
    return result

# 创建 Gradio 接口
interface = gr.Interface(fn=save_image_and_run, inputs=["video", "image"], outputs="video")
interface.launch()

if __name__ == "__main__":
    main()
