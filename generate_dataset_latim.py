import trimesh
import os
from os import listdir
from os.path import isfile, join
import random
import numpy as np
from scipy.spatial.transform import Rotation
import PIL.Image
import sys
import time
import logging
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)
import argparse
import glob

parser = argparse.ArgumentParser(description='create a synthetic dataset with the files found in the folder ')
parser.add_argument('--start', type=int, help='start point (default: 0)', default=0)
parser.add_argument('--samples', type=int, help='number of scenes to create')
parser.add_argument('--resolution', type=list, help='resolution of the depth maps and pointcloud projection (default: [640,480])', default=[640,480])
args = parser.parse_args()

def render_segmentation_images(meshes, resolution):
    """Renders segmentation masks (modal and amodal) for each object in the state.
    """

    full_depth = self.render_camera_image(color=False)
    amodal_data = np.zeros((full_depth.shape[0], full_depth.shape[1], len(self.obj_keys)), dtype=np.uint8)
    renderer = OffscreenRenderer(resolution[1], resolution[0])
    flags = RenderFlags.DEPTH_ONLY

    # Hide all meshes
    obj_mesh_nodes = [next(iter(self._scene.get_nodes(name=k))) for k in self.obj_keys]
    for mn in meshes.nodes_geometry:
        mn.mesh.is_visible = False

    for i, node in enumerate(obj_mesh_nodes):
        node.mesh.is_visible = True

        depth = renderer.render(self._scene, flags=flags)
        amodal_mask = depth > 0.0
        amodal_data[amodal_mask,i] = np.iinfo(np.uint8).max
        node.mesh.is_visible = False

    renderer.delete()
    
    # Show all meshes
    for mn in self._scene.mesh_nodes:
        mn.mesh.is_visible = True

    return amodal_data, modal_data
#Returns the depth image and the projected pointcloud from camera view of the scene
def gen_depth_and_pointcloud_rays(meshes, resolution=[960,600],show=False):
    # any of the automatically generated values can be overridden
    # set resolution, in pixels
    scene=meshes.scene()
    scene.camera.resolution = resolution
    # set field of view, in degrees
    # make it relative to resolution so pixels per degree is same
    #scene.camera.fov = 60 * (scene.camera.resolution /
    #                         scene.camera.resolution.max())
    scene.camera.fov = [39,25]
    # convert the camera to rays with one ray per pixel
    origins, vectors, pixels = scene.camera_rays()

    # do the actual ray- mesh queries
    pye = trimesh.ray.ray_pyembree.RayMeshIntersector(meshes)
    points, index_ray, index_tri = pye.intersects_location(
        origins, vectors, multiple_hits=False)

    
    # for each hit, find the distance along its vector
    depth = trimesh.util.diagonal_dot(points - origins[0],
                                      vectors[index_ray])
    # find pixel locations of actual hits
    pixel_ray = pixels[index_ray]

    # create a numpy array we can turn into an image
    # doing it with uint8 creates an `L` mode greyscale image
    a = np.zeros(scene.camera.resolution, dtype=np.uint8)

    # scale depth against range (0.0 - 1.0)
    depth_float = ((depth - depth.min()) / depth.ptp())

    # convert depth into 0 - 255 uint8
    depth_int = (depth_float * 255).round().astype(np.uint8)
    # assign depth to correct pixel locations
    a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int

    # create a PIL image from the depth queries
    depth_map = PIL.Image.fromarray(np.transpose(a))
    depth_map = depth_map.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    # show the resulting image
    if(show==True):
        depth_map.show()
    
    return depth_map

def clear_target_directories():
    files = glob.glob('./depth/*')
    for f in files:
        os.remove(f)
    files = glob.glob('./depth_ims/*')
    for f in files:
        os.remove(f)

def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])
def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])
def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])
def main():
    samples = args.samples
    start = args.start
    resolution = args.resolution

    clear_target_directories()

    print("---------  Generating %d scenes----------------"%(samples-start))
    #dir_path= './pc/' #ply files
    dir_path= './objects/' #stl files
    onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    print("%d different objects"%len(onlyfiles))
    stats = {}
    for i in range(start, start+samples):     
        print("Generating %06d"%i)
        meshes = trimesh.Trimesh()
        rand = random.randrange(3,7)
        for p in range(rand):
            #Load random 3D model
            random.shuffle(onlyfiles)
            pc= open(dir_path+'%s'%onlyfiles[p], 'r')
            geometry = trimesh.load(pc.name, file_type='stl')
            name = onlyfiles[p].split('.')[0]
            if name in stats:
                stats[name] +=1
            else:
                stats[name] = 1
            angle_x = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
            angle_y = (np.random.random()*np.pi/3) - np.pi/6
            angle_z = (np.random.random()*np.pi/3) - np.pi/6
            Rx = rotx(angle_x)
            Ry = roty(angle_y)
            Rz = rotz(angle_z)
            l,w,h = geometry.extents
            R = np.dot(Rz, np.dot(Ry,Rx))
            t =np.random.rand(1,3)                                                  #create a random vector of shape (1,3) with values between [0,1[
            trans = (t-0.5)                                                     #translation vector shifted and centered between [-2,2[                  

            M_rot_trans= np.zeros((4,4))       # Translation and Rotation
            M_rot_trans[:3,:3]= R
            M_rot_trans[:3,3]=trans
            M_rot_trans[3,:]= [0,0,0,1]    
            
            geometry.apply_transform(M_rot_trans)

            #add the object to the scene
            meshes = trimesh.util.concatenate( [ meshes, geometry ] )

        random_objects= random.randrange(2,5)
        for j in range(random_objects):
            t =np.random.rand(1,3)
            trans = (t-0.5)*2           
            rot = Rotation.random().as_matrix()           
            M= np.zeros((4,4))       # Translation and Rotation
            M[:3,:3]=rot
            M[:3,3]=trans
            M[3,:]= [0,0,0,1]
            box = trimesh.creation.box([random.random()/2, random.random()/2, random.random()/2], M)
            meshes = trimesh.util.concatenate( [ meshes, box ] ) 
        resolution=[960,600]
        depth_map= gen_depth_and_pointcloud_rays(meshes, resolution)
        
        depth_map.save("./depth_ims/%06d.jpg"%i)
        img = meshes.scene().save_image(resolution=resolution, show=True)
        with open("./images/%06d.png"%i, 'wb') as f:
            f.write(img)
            f.close()

    with open('stats.txt', 'w') as f:
        import collections
        stats = collections.OrderedDict(sorted(stats.items()))
        for k, v in stats.items():
            print("%s: %s"%(k,v), file=f)
    

if __name__ == '__main__':
    tic = time.perf_counter()
    main()
    toc = time.perf_counter()
    print("Generating dataset took %f seconds"%(toc-tic))