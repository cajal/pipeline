from pipeline import pupil
import os

for case in key_tests[0:3]:
    source_vid = (pupil.Eye & case).get_video_path()
    orig_dir = os.path.dirname(source_vid)
    file_name = os.path.basename(source_vid)
    hardlink_vid = os.path.join(orig_dir, file_name.split('.')[0] + '_tracking', file_name)
    
    files = os.listdir(orig_dir, file_name.split('.')[0] + '_tracking')
    
    if file_name in files:
        os.remove(file_name)
    
    os.link(source_vid, hardlink_vid)
