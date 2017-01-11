import pathlib
import glob
import os




def rename_files(image_dir):
	image_dir_path=pathlib.Path(image_dir)
	for image_path in (image_dir_path.glob('*')):
		if image_path.name[2]=='_' or image_path.name[2]=='a':
			new_name=image_path.name[0]+'0'+image_path.name[1:]
			new_name=image_dir+'/'+new_name
			os.rename(image_dir+'/'+image_path.name, new_name)
		if image_path.name[0]=='.':    
			os.unlink(image_dir+'/'+image_path.name)
