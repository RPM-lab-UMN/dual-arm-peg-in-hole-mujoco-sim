# import pybullet as p
import glob
import os
import tqdm
import pymeshlab
import numpy as np
import transforms3d as t3d


if __name__ == '__main__':
    in_objs = glob.glob('./*.obj')
    for in_obj in in_objs:
        if in_obj.endswith('_preprocessed_convex.obj'):
            os.remove(in_obj)
    in_objs = filter(lambda s: not s.endswith('_preprocessed.obj'), in_objs)
    in_objs = filter(lambda s: not s.endswith('_preprocessed_convex.obj'), in_objs)
    in_objs = sorted(in_objs)
    print(in_objs)

    # polygon -> triangle
    for in_obj in tqdm.tqdm(in_objs, desc='polygon -> triangle'):
        print(in_obj)
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(in_obj)
        # ms.apply_filter('set_color_per_vertex', color1=pymeshlab.Color(255, 0, 0))
        # ms.apply_filter('compute_color_transfer_mesh_to_face')
        # T = np.eye(4)
        # T[:3, :3] = t3d.euler.euler2mat(np.pi / 2, 0, 0)
        # ms.apply_filter('set_matrix', transformmatrix=T)
        if 'U_cube_cap' in in_obj:
            print('turn 180 around x axis')
            T = np.eye(4)
            T[:3, :3] = t3d.euler.euler2mat(0, np.pi, 0)
            ms.apply_filter('set_matrix', transformmatrix=T)
        ms.apply_filter('compute_normal_per_face')
        ms.apply_filter('compute_normal_per_vertex')
        out_obj = in_obj.replace('.obj', '_preprocessed_convex.obj')
        # ms.save_current_mesh(out_obj, save_polygonal=False)#, save_face_color=True, save_vertex_color=True, save_vertex_normal=False)
        ms.save_current_mesh(out_obj, save_polygonal=False, save_vertex_normal=False)

    # convex decompose
    # in_objs = [s.replace('.obj', '_preprocessed_convex.obj') for s in in_objs]
    # log_file = 'decompose_log.txt'
    # p.connect(p.DIRECT)
    # for in_obj in tqdm.tqdm(in_objs, desc='convex decompose'):
    #     print(in_obj)
    #     out_obj = in_obj
    #     p.vhacd(in_obj, out_obj, log_file)