## change pending image/node-affinity
from kubernetes import client, config 

config.load_kube_config()
v1 = client.CoreV1Api()

namespace = 'default'
selector = 'status.phase=Pending'
new_image = 'at-docker:5000/pipeline:at-node'
node_list = [f"at-node{num}" for num in (1,51)]
at_compute = [f"at-compute00{i}" for i in range(2,7)]
affinity_body = {
    'required_during_scheduling_ignored_during_execution':{
        'node_selector_terms':[{
            'matchExpressions':[{
                'key':'kubernetes.io/hostname',
                 'operator':'In',
                 'values':node_list + at_compute
                }]
            
        }]
    
    }
}

pending = v1.list_namespaced_pod(namespace=namespace,field_selector=selector)
for pod in pending.items():
    pod.spec.containers[0] = new_image 
    pod.spec.affinity.node_affinity = affinity_body
    response = v1.patch_namespaced_pod(
        name=pod.metadata.name,
        namespace='default',
        body = pod
    ) 





