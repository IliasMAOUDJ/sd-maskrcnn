from __future__ import division
import trimesh
import os
from os import listdir
from os.path import isfile, join
import random
from scipy.spatial.transform import Rotation
import numpy as np
import time
import PIL.Image

def gen_labels(path, scene, scene_number):
    dir_path= path+"labels/"
    if not os.path.exists(dir_path):
            os.mkdir(dir_path)
    copy_node = scene.geometry.copy()
    labels=[]
    for geometry in copy_node:
        if(geometry=="wall" or geometry=="floor" or "box" in geometry):
            continue
        
        if("Femur" in geometry):
            labels.append(1)
        elif("Tibia" in geometry):
            labels.append(2)
    np.save(dir_path+"image_%06d.npy"%(scene_number), labels)

def gen_semantic_data(path, scene, scene_number,resolution=[512,512]):
    scene.camera.resolution = resolution
    scene.camera.fov = 60 * (scene.camera.resolution /
                             scene.camera.resolution.max())
    origins, vectors, pixels = scene.camera_rays()
  
    dir_path= path+"semantic_masks/"
    if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    copy_node = scene.geometry.copy()
    removed_hidden_objects_scene =scene.copy()
    s=1
    arr_geom = []
    semantic = np.zeros(scene.camera.resolution, dtype=np.uint8)
    depth_map = np.full(scene.camera.resolution,255, dtype=np.uint8)
    for geometry in copy_node:
        #only bones
        if(geometry=="wall" or geometry=="floor"):
            continue
        copy_scene = scene.copy()
        for other in copy_node:
            if (other != geometry):
                copy_scene.delete_geometry(other)
        dump = copy_scene.dump(concatenate=True)
        pye = trimesh.ray.ray_pyembree.RayMeshIntersector(dump)
        points, index_ray, index_tri = pye.intersects_location(
            origins, vectors, multiple_hits=False)
        # for each hit, find the distance along its vector
        
        depth = trimesh.util.diagonal_dot(points - origins[0],
                                        vectors[index_ray])

        # find pixel locations of actual hits
        pixel_ray = pixels[index_ray]
        # convert depth into 0 - 255 uint8
        depth_float = ((depth - depth.min()) / depth.ptp())
        depth_int = (depth * 255).round().astype(np.uint8)
        
        e=0
        for (i,j) in pixel_ray:              #another solution to try is compute full depth map and then compare if the value is below threshold with numpy (takes less lines)
            if(depth_int[e] < depth_map[i,j]):
                if("box" in geometry):
                    semantic[i,j] = 0               
                else:  
                    semantic[i,j] = s                  
                depth_map[i,j] = depth_int[e]               
            e+=1
        if("box" not in geometry):
            arr_geom.append(geometry)
            s+=1
    # create a PIL image from the depth queries
    semantic_map = PIL.Image.fromarray(np.transpose(semantic))
    pixel_values = list(semantic_map.getdata())

    arr = np.zeros(s-1, dtype =np.bool)
    for j in np.arange(1,arr.size+1):
        if j in pixel_values:
            arr[j-1]=True
    for k in np.arange(arr.size):
        if(arr[k]==False):
            scene.delete_geometry(arr_geom[k])
            print("an object was removed because completely hidden or not enough pixels")
    semantic_map = semantic_map.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    file_name = dir_path+"image_%06d.png"%(scene_number)
    semantic_map.save(file_name)
    return scene
def gen_depth_image(path, scene, scene_number,resolution=[512,512]):
    # any of the automatically generated values can be overridden
    # set resolution, in pixels
    scene.camera.resolution = resolution
    # set field of view, in degrees
    # make it relative to resolution so pixels per degree is same
    scene.camera.fov = 60 * (scene.camera.resolution /
                             scene.camera.resolution.max())
    # convert the camera to rays with one ray per pixel
    origins, vectors, pixels = scene.camera_rays()

    dump = scene.dump(concatenate=True)
    # do the actual ray- mesh queries
    pye = trimesh.ray.ray_pyembree.RayMeshIntersector(dump)
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
    
    dir_path= path+"depth_ims/"
    if not os.path.exists(dir_path):
            os.mkdir(dir_path)
    file_name =dir_path+"/image_%06d.png"%(scene_number)
    depth_map.save(file_name)

    return a

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

from shutil import rmtree
if __name__ == '__main__':
    print("---------  Generating dataset scenes----------------")
    nb_samples=20
    
    train_inds = []
    test_inds = []
    total_pixels =[] #used for computing mean pixel
    stats = {'Femur':0, 'Tibia': 0}
    
    TRAIN_TEST_SPLIT = 0.9

    cam_trns= np.eye(4,4)
    cam_trns[:3,3]=[0,0,1.5]
    cam_trns[3,:]= [0,0,0,1]  
    dataset_path = "./tester/"
    if os.path.exists(dataset_path):
        rmtree(dataset_path)
    dataset_path+="images/"
    os.makedirs(dataset_path)
            
    for i in range(nb_samples):
        if i < int(nb_samples*TRAIN_TEST_SPLIT):
            dir_path= './datasets/objects/meshes/latim_train/' #stl files     
        else: 
            dir_path= './datasets/objects/meshes/latim_test/'
        onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
        print("Generating %06d"%i)
        #meshes = trimesh.Trimesh()

        resolution=[512,512]
        cam = trimesh.scene.Camera("cam", resolution=resolution, fov=[39,25])
        scene = trimesh.Scene(camera=cam, camera_transform=cam_trns)      

        M_rot_trans= np.eye(4,4)
        M_rot_trans[:3,3]=[0,-0.6,0]
        M_rot_trans[3,:]= [0,0,0,1]
        floor = trimesh.creation.box([10,0.05,10], M_rot_trans)
        scene.add_geometry(floor, geom_name="floor")

        M_wall= np.eye(4,4)
        M_wall[:3,3]=[0,4,-0.6]
        M_wall[3,:]= [0,0,0,1]
        wall = trimesh.creation.box([10,10,0.05], M_wall)
        scene.add_geometry(wall, geom_name="wall")

        rand = random.randrange(3,5)
        for p in range(rand):
            #Load random 3D model
            obj = random.choice(onlyfiles)
            pc= open(dir_path+'%s'%obj, 'r')
            geometry = trimesh.load(pc.name, file_type='stl')
            name = obj.split('.')[0]
            angle_x = (np.random.random()*np.pi/2) - np.pi/4 # -45 ~ +45 degree
            angle_y = (np.random.random()*np.pi/2) - np.pi/4
            angle_z = (np.random.random()*2*np.pi)
            Rx = rotx(angle_x)
            Ry = roty(angle_y)
            Rz = rotz(angle_z)
            R = np.dot(Rz, np.dot(Ry,Rx))
            t =np.random.rand(1,3)                                                  #create a random vector of shape (1,3) with values between [0,1[
            trans = (t-0.5)                                                     #translation vector shifted and centered between [-2,2[                  

            M_rot_trans= np.zeros((4,4))       # Translation and Rotation
            M_rot_trans[:3,:3]= R
            M_rot_trans[:3,3]=trans
            M_rot_trans[3,:]= [0,0,0,1]    
            
            geometry.apply_transform(M_rot_trans)
            #add the object to the scene
            scene.add_geometry(geometry, geom_name=name)
            if "Femur" in name:
                stats['Femur']+=1
            elif "Tibia" in name:
                stats['Tibia']+=1
            if name in stats.keys():
                stats[name]+=1
            else:
                stats[name]=1

        random_objects= random.randrange(2,5)
        for j in range(random_objects):
            t =np.random.rand(1,3)
            trans = (t-0.5)           
            rot = Rotation.random().as_matrix()           
            M= np.zeros((4,4))       # Translation and Rotation
            M[:3,:3]=rot
            M[:3,3]=trans
            M[3,:]= [0,0,0,1]
            obj = random.randint(1,2)
            if(obj==1):         
                geom = trimesh.creation.box(extents=[random.random()/2, random.random()/2, random.random()/2], transform=M)
            elif(obj==2):
                geom = trimesh.creation.cylinder(radius=random.random()/4, height= random.random()/3, transform=M)
            scene.add_geometry(geom, geom_name="box_%d"%j)

        scene = gen_semantic_data(dataset_path, scene, i, resolution)
        gen_labels(dataset_path, scene,i)
        pixels = gen_depth_image(dataset_path, scene,i,resolution)

        total_pixels.append(pixels)
        # Save split
        if i < int(nb_samples*TRAIN_TEST_SPLIT):
            train_inds.append(i)
        else:
            test_inds.append(i)

    mean = np.mean(total_pixels)
    np.save(os.path.join(dataset_path, 'train_indices.npy'), train_inds)
    np.save(os.path.join(dataset_path, 'test_indices.npy'), test_inds)  
    
    with open(dataset_path+'stats.txt', 'w') as f:
        import collections
        stats = collections.OrderedDict(sorted(stats.items()))
        for k, v in stats.items():
            print("%s: %s"%(k,v), file=f)