from tqdm import tqdm
import os
from renderer import BrepRenderer
import mitsuba as mi
from PIL import Image
import numpy as np
import h5py as h5


if __name__ == '__main__':
    part_paths = [d.path for d in os.scandir('parts')]

    height = 256
    width = 256
    viewpoint = [-0.25, -1.8, 1.125]
    target = [0, 0, 0]
    total_size = 10000
    cache_path = 'cache'
    renderer = BrepRenderer(width, height)

    if not os.path.isdir(cache_path):
        os.makedirs(cache_path)
    h5file = h5.File(os.path.join(cache_path, 'aovs.hdf5'), 'w')
    depth_dset = h5file.create_dataset('depth_raw', (total_size, height, width), dtype=np.float32)
    index_dset = h5file.create_dataset('primitives_raw', (total_size, height, width), dtype=np.int32)
    obj_dset = h5file.create_dataset('objects_raw', (total_size, height, width), dtype=np.float32)


    for idx in tqdm(range(total_size)):
        mesh_filename = part_paths[idx % len(part_paths)]
        
        image_path = os.path.join(cache_path, f'{idx}.png')
        normal_path = os.path.join(cache_path, f'{idx}_normals.png')
        renderer.initialize_scene(mesh_filename)
        output = renderer.render(viewpoint = viewpoint, target = target, depth=True, normals=True, primitives=True)

        mi.util.write_bitmap(image_path, output['rgb'])
        if 'aovs' in output:
            index_map = output['aov_index_map']
            aov_img = output['aovs']
            if 'depth' in index_map:
                depth_dset[idx] = aov_img[:,:,index_map['depth']]
            if 'prim' in index_map:
                index_dset[idx] = aov_img[:,:,index_map['prim']]
            if 'shape' in index_map:
                obj_dset[idx] = aov_img[:,:,index_map['shape']]
            if 'norm' in index_map:
                normal_map = np.array(aov_img[:,:,index_map['norm']:index_map['norm']+3])
                normal_map = normal_map * 0.5 + 0.5
                normal_map = np.round(np.clip(normal_map * 255, 0, 255)).astype(np.uint8)
                img = Image.fromarray(normal_map)
                img.save(normal_path)
