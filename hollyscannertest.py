import numpy
import re
import subprocess
import zbar
import freeimage
import os.path
import datetime
import time
import pathlib
from scanner_lifespans import run_analysis

# bottom portion whole plate: scanimage -l 35 -t 168 -x 130 -y 297 --resolution=500 --brightness=3 --mode=Gray --depth=8 --source 'Transparency Unit' --film-type 'Positive Film' --format=tiff>image1.tiff 
# top portion whole plate: scanimage -l 33 -t 84 -x 130 -y 75 --resolution=300 --brightness=5 --mode=Gray --depth=8 --source 'Transparency Unit' --film-type 'Positive Film' --format=tiff>image1.tiff 
# bottom portion qr code: scanimage -l 35 -t 170 -x 11 -y 11 --resolution=800 --brightness=5 --mode=Gray --depth=8 --source 'Transparency Unit' --film-type 'Positive Film' --format=tiff>image1.tiff
# top portion qr code: scanimage -l 33 -t 84 -x 11 -y 11 --resolution=300 --brightness=5 --mode=Gray --depth=8 --source 'Transparency Unit' --film-type 'Positive Film' --format=tiff>image1.tiff 

scanner_name_list='ABCDEFGHIJJK'
SANE_PGM_REGEX = re.compile(b'''
    P5[\n]           # start with literal text 'P5' and a newline
    [#].+[\n]        # then a comment line
    (\d+)[ ](\d+)[\n]  # then width and height separated by a space
    (\d+)[\n]        # then maxval
    ''', flags=re.VERBOSE)

def convert_to_8bit(image):
    test_image=[]

    for sublist in image:
        test_image.append(numpy.asarray(sublist))
      

    new_image=[]

    for sublist in test_image:
        image_type=str(sublist.dtype)
        if image_type=='>u2':
            sublist=(sublist/256).astype('uint8')
            new_image.append(sublist)
        else:
            new_image.append(sublist)
              
    return(new_image)

def get_scanners():
    
    
    out=subprocess.check_output(['scanimage', '-L'])
    out=out.decode("utf-8")
    available_scanners=re.findall('epson.+[0-9][0-9][1-9]...[0-9]', out)

    scanner_command_dict={}
    scanner_qr_command_dict = {}
    scanner_alphaname_dict={}
    strain_qr_command_dict={}
    strain_name_dict={}


    for i in range(len(available_scanners)):
        
        scanner_qr_command_dict[i]=['scanimage', '-d', available_scanners[i], '-p', '--mode=Gray','--depth=8', '--resolution=800', '--brightness=5', '-l 0', '-t 260', '-x 30', '-y 30']
        
    

    scanner_id_images =multi_scan(scanner_qr_command_dict)
    
    scanner_id_images=convert_to_8bit(scanner_id_images)
    
   
    

    


    scanner_object=zbar.Scanner()

    print("Available scanners are:")

    for i, sublist in enumerate(scanner_id_images):
        qr_string=str(scanner_object.scan(sublist)) 
        scanner_qr_name=re.search('scanner_[A-Z]', qr_string)
        scanner_qr_name=scanner_qr_name.group()
        scanner_alphaname_dict[i] = scanner_qr_name
        print(scanner_alphaname_dict[i])

    if scanner_alphaname_dict[0]=='scanner_A':
        scanner_command_dict[0]=['scanimage', '-d', available_scanners[0], '-p', '--mode=Gray','--depth=8', '--resolution=2400', '-l 35', '-t 65', '-x 80', '-y 120', '--source', 'Transparency Unit', '--film-type', 'Positive Film', '--brightness=3']
        strain_qr_command_dict[0] = ['scanimage', '-d', available_scanners[0], '-p', '--mode=Gray','--depth=8', '--resolution=800', '--brightness=5', '-l 70', '-t 175', '-x 20', '-y 20']
    
    if scanner_alphaname_dict[0]=='scanner_B':
        scanner_command_dict[0]=['scanimage', '-d', available_scanners[0], '-p', '--mode=Gray','--depth=8', '--resolution=2400', '-l 33', '-t 84', '-x 130', '-y 75', '--source', 'Transparency Unit', '--film-type', 'Positive Film', '--brightness=3']  
        strain_qr_command_dict[0] = ['scanimage', '-d', available_scanners[0], '-p', '--mode=Gray','--depth=8', '--resolution=800', '--brightness=5', '-l 33', '-t 84', '-x 20', '-y 20']  

    if scanner_alphaname_dict[1]=='scanner_A':
        scanner_command_dict[1]=['scanimage', '-d', available_scanners[1], '-p', '--mode=Gray','--depth=8', '--resolution=2400', '-l 35', '-t 65', '-x 80', '-y 120', '--source', 'Transparency Unit', '--film-type', 'Positive Film', '--brightness=3']
        strain_qr_command_dict[1] = ['scanimage', '-d', available_scanners[1], '-p', '--mode=Gray','--depth=8', '--resolution=800', '--brightness=5', '-l 70', '-t 175', '-x 20', '-y 20']    
    
    if scanner_alphaname_dict[1]=='scanner_B':
       scanner_command_dict[1]=['scanimage', '-d', available_scanners[0], '-p', '--mode=Gray','--depth=8', '--resolution=2400', '-l 33', '-t 84', '-x 130', '-y 75', '--source', 'Transparency Unit', '--film-type', 'Positive Film', '--brightness=3']       
       strain_qr_command_dict[1] = ['scanimage', '-d', available_scanners[0], '-p', '--mode=Gray','--depth=8', '--resolution=800', '--brightness=5', '-l 33', '-t 84', '-x 20', '-y 20']  

    
    strain_id_images=multi_scan(strain_qr_command_dict)
    
    strain_id_images=convert_to_8bit(strain_id_images)    


    for i, sublist in enumerate(strain_id_images):
        qr_string=str(scanner_object.scan(sublist))   
        strain_name=re.search('\w\w\w-\d\d.*\d\d-\d\d-\d\d', qr_string)
        if strain_name is None:
            print('Failed to parse strain name from scanned qr data: \"{}\"'.format(qr_string))
        else:
            strain_name=strain_name.group()
            strain_name_dict[i] = strain_name
            print(strain_name + " is on " + scanner_alphaname_dict[i])

    return available_scanners, strain_name_dict, scanner_command_dict

def scan(iterations, available_scanners, strain_name_dict, scanner_command_dict, sleep):

    for i in range(0,iterations):
        
        date = datetime.datetime.today()
        standard_date = datetime.date.isoformat(date)
        time=datetime.datetime.now().time()
        standard_time=time.isoformat()
        name=standard_date + standard_time
        date = datetime.datetime.today()
        standard_date = datetime.date.isoformat(date)
        time=datetime.datetime.now().time()
        standard_time=time.isoformat()
        name=standard_date + standard_time
        print(i)
        print("Start: " + datetime.datetime.now().time().isoformat())

        plate_scan_filenames = {}
        paths = {}
        
        
        
        for j in range(len(available_scanners)):
            
            newpath = '/mnt/bulkdata/hkinser/Desktop/' + strain_name_dict[j]
            paths[j] = newpath
            if not os.path.exists(newpath):os.makedirs(newpath)
            plate_scan_filenames[j] = newpath+'/' +name+ strain_name_dict[j]  + '.tif'

        

        plate_images=multi_scan(scanner_command_dict)
        plate_images=convert_to_8bit(plate_images)
        
     

        for k, sublist in enumerate(plate_images):
           
            freeimage.write(sublist, plate_scan_filenames[k])
        


                
        print("End: " + datetime.datetime.now().time().isoformat())
        import time
        print(i)
        
        if sleep==True and i !=(iterations-1):
            print('sleeping')
            time.sleep(600)
        else:
            print('doing analysis')
            for l in range(0, len(strain_name_dict)):
                do_analysis(strain_name_dict[l])
            
def do_analysis(strain_name):
    in_dir='/mnt/bulkdata/hkinser/Desktop/'+strain_name
    out_dir=in_dir+'_OUT'
    old_lifespan_file=pathlib.Path(out_dir+'/lifespans.csv')
    old_lifespan_pickle=pathlib.Path(out_dir+'/lifespans.pickle')
    if pathlib.Path.exists(old_lifespan_pickle):
        pathlib.Path.unlink(old_lifespan_pickle)
    if pathlib.Path.exists(old_lifespan_file):
        pathlib.Path.unlink(old_lifespan_file)
        
    age=re.search('\ddph', in_dir)
    age=re.search('\d', age.group())
    age=(int)(age.group())
    list_of_training_data=['/mnt/bulkdata/hkinser/Desktop/mir-71 GFP 3dph 01-08-16_OUT/trainingdata.pickle','/mnt/bulkdata/hkinser/Desktop/mir-71 mcherry3 3dph 01-08-16_OUT/trainingdata.pickle','/mnt/bulkdata/hkinser/Desktop/mir-238 GFP 3dph 02-02-16_OUT/training_data.pickle', '/mnt/bulkdata/hkinser/Desktop/mir-240 GFP 4dph 02-11-16_OUT/trainingdata.pickle', '/mnt/bulkdata/hkinser/Desktop/mir-42 GFP 4dph 02-15-16_OUT/trainingdata.pickle']
    run_analysis.run_analysis(in_dir, out_dir, age, run_analysis.HOLLY_NAME_PARAMS, run_analysis.HOLLY_PLATE_PARAMS, run_analysis.HOLLY_IMAGE_SCORE_PARAMS, list_of_training_data, False, False)
    
    
    
            
       

def read_sane_pgm(pgm_bytes):
    match = SANE_PGM_REGEX.match(pgm_bytes)
    if match is None:
        raise ValueError('Could not read SANE-format PGM file.')
    width, height, maxval = map(int, match.groups())
    header_end = match.span()[1]
    dtype = numpy.uint8 if maxval < 256 else '>u2' # '>u2' means 2-byte unsigned, most significant byte first
    return numpy.ndarray(shape=(width, height), dtype=dtype, buffer=pgm_bytes, offset=header_end, order='F')

def multi_scan(scanner_command_dict):
    scan_processes = [subprocess.Popen(values, stdout=subprocess.PIPE) for values in scanner_command_dict.values()]
    images = []
    for i, process in enumerate(scan_processes):
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print('Problem with scan {} (command: {})')
            images.append(None)
        else:
            try:
                images.append(read_sane_pgm(stdout))
            except ValueError:
                print('Could not parse image from scan {} (command: {})')
                images.append(None)
    return images
