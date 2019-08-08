from commons import lab
from pipeline import experiment
import os
import datajoint as dj

pupil = dj.create_virtual_module('pipeline_eye', 'pipeline_eye')

def get_video_path(key):
    video_info = (experiment.Session() *
                  experiment.Scan.EyeVideo() & key).fetch1()
    video_path = lab.Paths().get_local_path(
        "{behavior_path}/{filename}".format(**video_info))
    return video_path

keys = pupil.FittedPupil.proj().fetch(as_dict=True)

for key in keys[0:3]:
    source_vid = get_video_path(key)
    orig_dir = os.path.dirname(source_vid)
    file_name = os.path.basename(source_vid)
    hardlink_vid = os.path.join(orig_dir, file_name.split('.')[0] + '_tracking', file_name)
    
    files = os.listdir(os.path.join(orig_dir, file_name.split('.')[0] + '_tracking'))
    
    if file_name in files:
        os.remove(file_name)
    
    os.link(source_vid, hardlink_vid)