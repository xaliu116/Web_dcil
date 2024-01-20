import first_order_motion.stage1 as stage1
import first_order_motion.stage2 as stage2
import os
import cv2



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
