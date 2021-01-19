import streamlit as st
import os
import cv2
import numpy as np
import searchconsole
from yolo_vid import Yolov3Video
from yolo_img import YoloImg
import tempfile
from autotrain import Yolov3Train

def main():
	st.title('Automated Object Detection with YOLOv3')
	menu = ['Images', 'Videos', 'Train']

	choice = st.sidebar.selectbox('Dev Menu',menu)

	if choice == 'Images':
		st.write('For Images')
		cfg = st.text_input('Path to cfg file')
		image = st.file_uploader('Select an image file', type=['jpg','jpeg'])
		names = st.text_input('Path to names file')
		weights = st.text_input('Path to weights file')
		submit = st.button('Run')
		print(weights)
		
		if submit:
			file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
			opencv_image = cv2.imdecode(file_bytes, 1)
			print(type(image))
			fn = image.name
			x = YoloImg(opencv_image, weights, cfg,names,fn)
			l = x.run_model()
			st.image(l,channels = "BGR",caption="Inference")	
	elif choice == 'Videos':
		st.write('For Videos')
		cfg_vid  = st.text_input('Path to cfg file')
		image_vid  = st.file_uploader('Select a video file', type=['mp4','mov'])
		
		names_vid  = st.text_input('Path to names file')
		weights_vid  = st.text_input('Path to weights file')
		submit_vid = st.button('Run')
		if submit_vid:
			print(image_vid.name)
			vn = image_vid.name
			tfile = tempfile.NamedTemporaryFile(delete=False)
			tfile.write(image_vid.read())
			vf = cv2.VideoCapture(tfile.name)
			x = Yolov3Video(vf, weights_vid,cfg_vid,names_vid,vn)
			for t in x.run_model():
				st.write(t)
				if '.mp4' in t:
					videofile = t
					print(videofile)
			m = os.path.splitext(videofile)[0]
			os.system('ffmpeg -i '+videofile+' -vcodec libx264 '+m+'fmpeg.mp4')
			st.video(m+'fmpeg.mp4')
	else:
		st.write('Automatic Training')
		labels = st.text_input('Path to labels/annotations folder')
		imgs = st.text_input('Path to images folder')
		classes = st.number_input('Number of Classes?')
		classnames = st.text_input('Array of Classes Eg:["Car","Bird"]')
		workdir = st.text_input('Path to working directory')
		subd = st.number_input('Number of Sub-Divisions?(Optional)')
		batches = st.number_input('Number of Batches?(Optional)')
		imgsize = st.number_input('Image Size?(Optional)')
		submit_train = st.button('Run')
		if submit_train:
			b = Yolov3Train(labels,imgs,classes,classnames,workdir)
			for i in b.train():
				st.write(i)




if __name__ == '__main__':
	main()