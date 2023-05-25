import mitsuba as mi
from mitsuba.scalar_rgb import Transform4f as T
from mitsuba.scalar_rgb import Matrix4f as M
from mitsuba.scalar_rgb import Matrix3f as M3f
mi.set_variant('cuda_ad_rgb')
import numpy as np

import drjit as dr
from copy import deepcopy

import trimesh


class BrepRenderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height


    def initialize_scene(self, mesh_filename, texture_filename='wildtextures-seamless-square-pattern-paper-texture.jpg', envmap_filename='studio_small_09_4k.exr', ground=True, unsmoothed_V = None, unsmoothed_F = None):
        self.unsmoothed_V = unsmoothed_V
        self.unsmoothed_F = unsmoothed_F
        scene_dict = {
            'type': 'scene',
            #'integrator': integrator,
            'environment': {
                'type': 'envmap',
                'filename': envmap_filename,
                'to_world': T.rotate(axis=[1, 0, 0], angle=90)

            },
            
            'mesh':{
                'type': 'obj',
                'filename': mesh_filename,
                'bsdf': {
                    'type': 'roughplastic',
                    'diffuse_reflectance': { 'type': 'rgb', 'value': (0.7, 0.7, 0.7) },
                },
            }
        }
        if ground:
            scene_dict['floor'] = {
                'type': 'cube',
                'to_world': T.scale([100, 100, 1.0]).translate([0, 0, -1]),
                'bsdf': {
                    'type': 'diffuse',
                    #'reflectance': { 'type': 'rgb', 'value': (0.7, 0.85, 0.9)},
                    'reflectance': { 'type': 'bitmap',
                                    'filename': texture_filename,
                                    'to_uv': T.scale([100, 100, 1])}
                },
            }
        self.scene = mi.load_dict(scene_dict)

        
    
    def render(self, viewpoint=[-2, 6, 4.5], target=[0, 0, 0], depth=False, primitives=False, normals=False):
        integrator = mi.load_dict({
            'type': 'direct',
        })
        sensor_rgb = mi.load_dict(
            {
                'type': 'perspective',
                'to_world': T.look_at(
                                origin=viewpoint,
                                target=target,
                                up=(0, 0, 1)
                            ),
                'fov': 60,
                'film': {
                    'type': 'hdrfilm',
                    'width': self.width,
                    'height': self.height,
                    'rfilter': { 'type': 'gaussian', 'stddev': 0.5 },
                    'sample_border': True
                },
            }
        )
        
        image_rgb = mi.render(self.scene, spp=1024, integrator=integrator, sensor=sensor_rgb)
        outputs = {'rgb': image_rgb}
        
        aovs = {}
        aov_index = 3
        if primitives:
            aovs['prim'] = ('prim_index', aov_index)
            aov_index += 1
            aovs['shape'] = ('shape_index', aov_index)
            aov_index += 1
        if normals:
            aovs['norm'] = ('geo_normal', aov_index)
            aov_index += 3
        if depth:
            aovs['depth'] = ('depth', aov_index)
            aov_index += 1
        
        if aovs:
            sensor_tris = mi.load_dict(
                {
                    'type': 'perspective',
                    'to_world': T.look_at(
                                    origin=viewpoint,
                                    target=target,
                                    up=(0, 0, 1)
                                ),
                    'fov': 60,
                    'film': {
                        'type': 'hdrfilm',
                        'width': self.width,
                        'height': self.height,
                        'rfilter': {'type': 'box'},
                        'sample_border': True
                    },
                })
            integrator_tris = mi.load_dict(
                {
                    'type': 'aov',
                    'aovs': ','.join([f'{aov}:{aovs[aov][0]}' for aov in aovs])
                }
            )
            image_prims = mi.render(self.scene, integrator=integrator_tris, sensor=sensor_tris, spp=1)
            outputs['aovs'] = image_prims
            outputs['aov_index_map'] = {aov: aovs[aov][1] for aov in aovs}

        return outputs
