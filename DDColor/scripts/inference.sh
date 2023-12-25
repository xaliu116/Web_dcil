CUDA_VISIBLE_DEVICES=0 \
python inference/colorization_pipline.py \
	--input ./test_imgs --output ./colorize_output \
	--model_path modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt