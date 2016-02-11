import datajoint as dj

# import pandas as pd
schema = dj.schema('pipeline_pupiltracking', locals())
from . import rf
import numpy as np
import os


@schema
class VideoGroup(dj.Lookup):
    definition = """
    # table that groups videos into groups that can be tracked by the same SVM
    videogroup_id       : tinyint # id of the video group
    ---
    group_name          : char(20) # name of the group
    """

    contents = [  # these contents will be automatically inserted into the database
        (1, 'setup_S505')
    ]


@schema
class SVM(dj.Lookup):
    definition = """
    # table that stores the paths for the SVMs for each VideoGroup
    ->VideoGroup
    version         : int   # version of the SVM
    ---
    svm_path        : varchar(200) # path to the SVM file
    """

    contents = [
        (1, 0, 'no_SVM'),
        (1, 1, '/media/lab/users/jagrawal/global_svm/svm_version1/svm'),
        (1, 2, '/media/lab/users/jagrawal/global_svm/svm_version2/svm'),
        (1, 3, '/media/lab/users/jagrawal/global_svm/svm_version3/svm'),
    ]


@schema
class EyeFrame(dj.Computed):
    definition = """
    # eye tracking info for each frame of a movie
    -> rf.Eye
    -> SVM
    frame                       : int                           # frame number in movie
    ---
    eye_frame_ts=CURRENT_TIMESTAMP    : timestamp               # automatic
    """

    @property
    def populated_from(self):
        # return rf.Eye() * SVM() * VideoGroup().aggregate(SVM(), current_version='MAX(version)') & 'version=current_version'
        return rf.Eye() * SVM() * VideoGroup().aggregate(SVM(), current_version='MAX(version)') & 'version=0'

    def _make_tuples(self, key):
        print("Entered make tuples")
        print(key)
        svm_path = (SVM() & key).fetch1['svm_path']
        print(svm_path)

        # x_roi=900
        # y_roi=420
        # pass
        roi = (rf.Eye() & key).fetch1['eye_roi']
        print(roi)
        patch_size = 380
        y_roi = (roi[0][1] + roi[0][3]) / 2 - patch_size / 2
        x_roi = (roi[0][0] + roi[0][2]) / 2 - patch_size / 2
        x_roi = 275 - patch_size / 2
        y_roi = 675 - patch_size / 2
        print(x_roi, y_roi)
        efd = EyeFrame.EyeFrameDetected()

        # Code to do tracking

        # print(key)
        kk = key['animal_id']
        si = key['scan_idx']
        # svm="/media/lab/users/jagrawal/global_svm/svm_version2/svm"
        out = "/media/lab/users/jagrawal/global_svm/151123/m7199A9eyetracking/out"
        video = "m" + str(kk) + "A" + str(si) + "eyetracking"
        command = "find /media/scratch01/WholeCell/jake/* -name " + video + ".avi"
        # print(command)
        video_path = os.popen(command).read()
        video_path = video_path.strip(' \n\t')
        # print("video_path=",video_path)
        folder = video_path.split("/")[5]
        print(folder)
        debug = 0
        if len(video_path) != 0:
            print("do")
            if (os.path.exists(folder + "/" + video)) and debug == 0:
                print("Data already exists for " + folder + "/" + video)
            else:
                print("Making directory: " + folder + "/" + video)
                command = "mkdir -p " + folder + "/" + video + "/images"
                if debug == 0:
                    os.system(command)

                if (svm_path.find('no_SVM') + 1):
                    # print("if")
                    command = "cd " + folder + "/" + video + "; python2 /media/lab/users/jagrawal/Pupil-tracking/track_without_SVM.py " + str(
                        int(x_roi)) + " " + str(int(y_roi)) + " " + video_path + " -P " + str(
                        int(patch_size)) + "; cd ../.."
                else:
                    # print("else")
                    command = "cd " + folder + "/" + video + "; python2 /media/lab/users/jagrawal/Pupil-tracking/track.py " + out + " " + svm_path + " " + video_path + "; cd ../.."

                print("Running command :", command)
                if debug == 0:
                    # print(command)
                    os.system(command)

                # CODE to insert data after tracking
                # NOTE: Text parsing will change to pandas database
                for i, line in enumerate(open(str(folder + '/' + video + "/trace.txt"))):
                    key['frame'] = i + 1
                    self.insert1(key)
                    if 'NONE' not in line:
                        sub_key = dict(key)
                        words = line.split()
                        sub_key['pupil_x'] = float(words[0].split('=')[1])
                        sub_key['pupil_y'] = float(words[1].split('=')[1])
                        sub_key['pupil_r_minor'] = float(words[2].split('=')[1])
                        sub_key['pupil_r_major'] = float(words[3].split('=')[1])
                        sub_key['pupil_angle'] = float(words[4].split('=')[1])
                        sub_key['pupil_x_std'] = float(words[9].split('=')[1])
                        sub_key['pupil_y_std'] = float(words[10].split('=')[1])
                        sub_key['pupil_r_minor_std'] = float(words[11].split('=')[1])
                        sub_key['pupil_r_major_std'] = float(words[12].split('=')[1])
                        sub_key['pupil_angle_std'] = float(words[13].split('=')[1])
                        efd.insert1(sub_key)
        else:
            print("Video not found")



    class EyeFrameDetected(dj.Part):
        definition = """
        # eye frames with detected eye
        ->EyeFrame
        ---
        pupil_x                     : float                         # pupil x position
        pupil_y                     : float                         # pupil y position
        pupil_r_minor               : float                         # pupil radius minor axis
        pupil_r_major               : float                         # pupil radius major axis
        pupil_angle                 : float                         # angle of major axis vs. horizontal axis in radians
        pupil_x_std                 : float                         # pupil x position std
        pupil_y_std                 : float                         # pupil y position std
        pupil_r_minor_std           : float                         # pupil radius minor axis std
        pupil_r_major_std           : float                         # pupil radius major axis std
        pupil_angle_std             : float                         # angle of major axis vs. horizontal axis in radians
        """
