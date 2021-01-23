import git
import os
import shutil

def split(full_path_to_images,workingd):
	os.chdir(full_path_to_images)
	p = []
	for current_dir, dirs, files in os.walk('.'):
	    for f in files:
	        if f.endswith('.jpg'):
	            path_to_save_into_txt_files = full_path_to_images + '/' + f
	            p.append(path_to_save_into_txt_files + '\n')

	p_test = p[:int(len(p) * 0.15)]
	p = p[int(len(p) * 0.15):]
	os.chdir(workingd+"/darknet/data/")
	with open('train.txt', 'w') as train_txt:
	    for e in p:
	        train_txt.write(e)
	with open('test.txt', 'w') as test_txt:
	    for e in p_test:
	        test_txt.write(e)
classes = 1
classnames = ['gun']

class Yolov3Train:
	def __init__(self,labels,images,classes,classnames,workingd,subdivisions=8,batch=16,imgsize=608):
		self.labels = labels # Full data path
		self.images=images
		self.classes = classes
		self.classnames = classnames
		self.workingd = workingd
		self.subdivisions = subdivisions
		self.batch = batch
		self.imgsize = imgsize
	def train(self):
		if self.workingd[-1] == "/":
			self.workingd = self.workingd[:-1]
		else:
			pass

		os.chdir(self.workingd)
		git.Repo.clone_from('https://github.com/pjreddie/darknet', 'darknet')
		yield('Github Cloned')
		shutil.move(self.images,self.workingd+"/darknet/data/")
		shutil.rmtree(self.workingd+"/darknet/data/labels/")
		shutil.move(self.labels,self.workingd+"/darknet/data/")
		print('Data Transferred...Creating files')
		yield('Data Transferred...Creating files')
		os.chdir(self.workingd+"/darknet/data/")
		f = open('custom.names','w+')
		for i in range(len(self.classnames)):
			f.write(self.classnames[i]+"\n")
		f.close()
		print('Custom.names files created')
		yield('Custom.names file created')
		split(self.workingd+'/darknet/data/'+os.path.basename(self.images),self.workingd)
		print('Train and Test split done and registered')
		yield('Train and Test split done and registered')
		os.chdir(self.workingd+"/darknet/data/")
		f = open('custom.data','w+')
		f.write('classes = '+str(self.classes)+'\n')
		f.write('train  = data/train.txt'+'\n')
		f.write('test  = data/test.txt'+'\n')
		f.write('names  = data/custom.names'+'\n')
		f.write('backup = backup/')
		f.close()
		print('Data file created')
		yield('Data file created')
		numfilters = 3*(self.classes+5)
		with open(self.workingd+"/darknet/cfg/yolov3.cfg", 'r') as file:
			cfgdata=file.readlines()
		cfgdata[5] = 'batch='+str(self.batch)+"\n"
		cfgdata[7] = "width="+str(self.imgsize)+"\n"
		cfgdata[8] = "height="+str(self.imgsize)+"\n"
		cfgdata[602] = "filters="+str(numfilters)+"\n"
		cfgdata[609] = "classes="+str(self.classes)+"\n"
		cfgdata[688] = "filters="+str(numfilters)+"\n"
		cfgdata[695] = "classes="+str(self.classes)+"\n"
		cfgdata[775] = "filters="+str(numfilters)+"\n"
		cfgdata[782] = "classes="+str(self.classes)+"\n"
		os.chdir(self.workingd+"/darknet/cfg/")
		f = open('yolov3-custom.cfg','w+')
		f.writelines(cfgdata)
		f.close()
		print('All Set to Train')
		yield('All Set to Train')
		source_dir = self.workingd+"/"+"darknet/data/"+os.path.basename(self.labels)
		target_dir = self.workingd+"/"+"darknet/data/"+os.path.basename(self.images)
		    
		file_names = os.listdir(source_dir)
		    
		for file_name in file_names:
		    shutil.move(os.path.join(source_dir, file_name), target_dir)
		os.chdir(self.workingd+"/darknet/")
		os.system('make')
		print('Directory made...Downloading Pretrained weights')
		yield('Directory made...Downloading Pretrained weights')
		os.system('wget https://pjreddie.com/media/files/darknet53.conv.74')
		print('Training Starting')
		yield('Training Starting watch for outputs in working directory...This could take a couple of hours')
		os.system('./darknet detector train data/custom.data cfg/yolov3-custom.cfg darknet53.conv.74')
		print('Training stopped or completed...Weights files should be in the backup folder')
		yield('Training stopped or completed...Weights files should be in the backup folder')
