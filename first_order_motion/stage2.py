import imageio
from first_order_model_master.demo import load_checkpoints, make_animation
from skimage import img_as_ubyte
from skimage.transform import resize
import warnings
import os

warnings.filterwarnings("ignore")

mydir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
temp = mydir + "//first_order_motion"
mydir = os.chdir(temp)

generator, kp_detector = load_checkpoints(
    config_path='first_order_model_master/config/vox-256.yaml',
    checkpoint_path='first_order_model_master/checkpoint/vox-cpk.pth.tar'
)

def generate_output():
    source_image = imageio.imread("stage1_image.jpg")
    reader = imageio.get_reader("stage1_video.mp4")

    source_image = resize(source_image, (256, 256))[..., :3]

    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

    predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

    # 生成视频的本地保存路径
    local_file_path = os.path.abspath('./generated.mp4')
    imageio.mimsave(local_file_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)

    # 返回生成的视频的本地绝对路径
    return local_file_path
