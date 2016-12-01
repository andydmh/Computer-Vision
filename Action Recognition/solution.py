from utils import *
from sys import *

#pass as an argument the path to the ucf action database directory

ucf_action_dir_path = argv[1]
print get_all_video_paths(ucf_action_dir_path)


