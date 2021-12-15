#!/usr/local/bin/python3

from kubernetes import client,config
import time
import datajoint as dj

config.load_incluster_config()
v1 = client.CoreV1Api()

experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')
treadmill = dj.create_virtual_module('treadmill', 'pipeline_treadmill')
pupil = dj.create_virtual_module('pupil', 'pipeline_pupil')
fuse = dj.create_virtual_module('fuse', 'pipeline_fuse')


while True:

    unfinished_treadmill = experiment.MesoClosedLoop - treadmill.Treadmill
    unfinished_pupil = experiment.MesoClosedLoop - pupil.Eye
    unfinished_fuse = experiment.MesoClosedLoop - fuse.ScanDone
    nodes = ['at-compute003','at-compute004','at-compute005']
    nodes = v1.list_node(label_selector=f'kubernetes.io/hostname in ({",".join(nodes)})')
        
    if unfinished_treadmill or unfinished_pupil or unfinished_fuse:
        for node in nodes.items:
            body = {
                'spec': {
                    'taints':[{
                        'key':'meso_closed_loop',
                        'operator':'Equal',
                        'value':'true',
                        'effect':'NoExecute'
                    }]
                }
            }
            v1.patch_node(node.metadata.name,body)
        
    else:
        for node in nodes.items:

            if node.spec.taints is not None:
                node.spec.taints = [taint for taint in node.spec.taints if not taint.key == 'meso_closed_loop']
            v1.patch_node(node.metadata.name,node)

    time.sleep(60)
