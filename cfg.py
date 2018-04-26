#image pré-processing
input_size = 416 #width=height
channels = 3 #RBG 
angle = 0
saturation = 1.5
exposure = 1.5
hue = 0.1 
jitter = 0.3
random = 1

batch_size = 30
threshold = 0.3
ignore_thresh = .5
truth_thresh = 1
momentum = 0.9
decay = 0.0005
learning_rate = 0.001
burn_in=1000
max_batches = 500200

#policy=steps
learning_rate_steps = [40000, 45000]	#steps=400000,450000
learning_rate_scales = [0.1,0.1]		#scales=.1,.1

#yolo 0,1,2 it means the number anchors per a layers is 3
anchors_012 = np.array([[10,13],[16,30],[33,23]]) 
#yolo 3,4,5
anchors_345 = np.array([[30,61],[62,45],[59,119]])
#yolo 6,7,8
anchors_678 = np.array([[116,90],[156,198],[373,326]])

num = 9 #9 anchors per grille celle
classes = 80

source_file = 'image' # or 'video'
video_name = 'test_video.mp4'
save_picture = False
show_picture = True