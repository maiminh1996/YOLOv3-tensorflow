batch_size = 30
input_size = 416
threshold = 0.3
channels =3

momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=1000
max_batches = 500200
policy=steps
steps=400000,450000
scales=.1,.1

source_file = 'image' # or 'video'
video_name = 'test_video.mp4'
save_picture = False
show_picture = True