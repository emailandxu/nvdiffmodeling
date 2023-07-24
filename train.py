# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import time
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

import src.renderutils as ru
from src import obj
from src import util
from src import mesh
from src import texture
from src import render
from src import regularizer
from src.mesh import Mesh
import random

from plyfile import PlyData, PlyElement

RADIUS = 3.5

# Enable to debug back-prop anomalies
# torch.autograd.set_detect_anomaly(True)

###############################################################################
# Utility mesh loader
###############################################################################

def load_mesh(filename, mtl_override=None):
    name, ext = os.path.splitext(filename)
    if ext == ".obj":
        return obj.load_obj(filename, clear_ks=True, mtl_override=mtl_override)
    assert False, "Invalid mesh file extension"

###############################################################################
# Loss setup
###############################################################################

def createLoss(FLAGS):
    if FLAGS.loss == "smape":
        return lambda img, ref: ru.image_loss(img, ref, loss='smape', tonemapper='none')
    elif FLAGS.loss == "mse":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='none')
    elif FLAGS.loss == "logl1":
        return lambda img, ref: ru.image_loss(img, ref, loss='l1', tonemapper='log_srgb', use_python=True)
    elif FLAGS.loss == "logl2":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='log_srgb')
    elif FLAGS.loss == "relativel2":
        return lambda img, ref: ru.image_loss(img, ref, loss='relmse', tonemapper='none')
    else:
        assert False

###############################################################################
# trainable setup
###############################################################################

class TrainableMesh():
    def __init__(self, FLAGS, base_mesh:Mesh, ref_mesh:Mesh) -> None:
        self.FLAGS = FLAGS
        self.base_mesh = base_mesh
        self.ref_mesh = ref_mesh

        # Create normalized size versions of the base and reference meshes. Normalized base_mesh is important as it makes it easier to configure learning rate.
        self.normalized_base_mesh = mesh.unit_size(self.base_mesh)

        self.opt_params = self.create_trainable_dict()
        self.trainable_list = self.create_trainable_list(**self.opt_params)

        self.opt_base_mesh, self.opt_detail_mesh = self.create_trainable_mesh(**self.opt_params)

        # ==============================================================================================
        #  Setup torch optimizer
        # ==============================================================================================
        self.optimizer  = torch.optim.Adam(self.trainable_list, lr=FLAGS.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda x: max(0.0, 10**(-x*0.0002))) 

    def create_trainable_dict(self):
        v_pos_opt = self.normalized_base_mesh.v_pos.clone().detach().requires_grad_(True)

        # Trainable normal map, initialize to (0,0,1) & make sure normals are always in positive hemisphere
        if self.FLAGS.random_textures:
            normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), self.FLAGS.texture_res, not self.FLAGS.custom_mip)
        else:
            if 'normal' not in self.ref_mesh.material:
                normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), self.FLAGS.texture_res, not self.FLAGS.custom_mip)
            else:
                normal_map_opt = texture.create_trainable(self.ref_mesh.material['normal'], self.FLAGS.texture_res, not self.FLAGS.custom_mip)

        # Setup Kd, Ks albedo and specular textures
        if self.FLAGS.random_textures:
            if self.FLAGS.layers > 1:
                kd_map_opt = texture.create_trainable(np.random.uniform(size=self.FLAGS.texture_res + [4], low=0.0, high=1.0), self.FLAGS.texture_res, not self.FLAGS.custom_mip)
            else:
                kd_map_opt = texture.create_trainable(np.random.uniform(size=self.FLAGS.texture_res + [3], low=0.0, high=1.0), self.FLAGS.texture_res, not self.FLAGS.custom_mip)

            ksR = np.random.uniform(size=self.FLAGS.texture_res + [1], low=0.0, high=0.01)
            ksG = np.random.uniform(size=self.FLAGS.texture_res + [1], low=self.FLAGS.min_roughness, high=1.0)
            ksB = np.random.uniform(size=self.FLAGS.texture_res + [1], low=0.0, high=1.0)
            ks_map_opt = texture.create_trainable(np.concatenate((ksR, ksG, ksB), axis=2), self.FLAGS.texture_res, not self.FLAGS.custom_mip)
        else:
            kd_map_opt = texture.create_trainable(self.ref_mesh.material['kd'], self.FLAGS.texture_res, not self.FLAGS.custom_mip)
            ks_map_opt = texture.create_trainable(self.ref_mesh.material['ks'], self.FLAGS.texture_res, not self.FLAGS.custom_mip)

        # Trainable displacement map
        displacement_map_var = None
        if self.FLAGS.subdivision > 0:
            displacement_map_var = torch.tensor(np.zeros(self.FLAGS.texture_res + [1], dtype=np.float32), dtype=torch.float32, device='cuda', requires_grad=True)


        # initalize trainable parameters for skining 
        bone_mtx_opt = None
        v_weights_opt = None
        if self.FLAGS.skinning:
            # frames, bones, 1/bones
            v_weights_opt = torch.ones_like(self.normalized_base_mesh.v_weights) / self.normalized_base_mesh.bone_mtx.shape[1]
            # copy initialization
            bone_mtx_opt = self.normalized_base_mesh.bone_mtx.clone().detach()
            bone_mtx_opt.requires_grad_(True)
            v_weights_opt.requires_grad_(True)
            
        return {
            "v_pos_opt" : v_pos_opt,
            "normal_map_opt" : normal_map_opt,
            "kd_map_opt" : kd_map_opt,
            "ks_map_opt" : ks_map_opt,
            "displacement_map_var" : displacement_map_var,
            "bone_mtx_opt": bone_mtx_opt,
            "v_weights_opt": v_weights_opt
        }

    def create_trainable_list(self, v_pos_opt, normal_map_opt, kd_map_opt, ks_map_opt, displacement_map_var, bone_mtx_opt, v_weights_opt, *args, **kwargs):
        # ==============================================================================================
        #  Initialize weights / variables for trainable mesh
        # ==============================================================================================
        trainable_list = [] 
        # Add trainable arguments according to config
        if not 'position' in self.FLAGS.skip_train:
            trainable_list += [v_pos_opt]        
        if not 'normal' in self.FLAGS.skip_train:
            trainable_list += normal_map_opt.getMips()
        if not 'kd' in self.FLAGS.skip_train:
            trainable_list += kd_map_opt.getMips()
        if not 'ks' in self.FLAGS.skip_train:
            trainable_list += ks_map_opt.getMips()
        if not 'displacement' in self.FLAGS.skip_train and displacement_map_var is not None:
            trainable_list += [displacement_map_var]

        # Add trainable arguments according to config
        if self.FLAGS.skinning and not 'bone_mtx' in self.FLAGS.skip_train: 
            trainable_list += [bone_mtx_opt]
        if self.FLAGS.skinning and not 'weights' in self.FLAGS.skip_train:
            trainable_list += [v_weights_opt]
        
        return trainable_list

    def create_trainable_mesh(self, v_pos_opt, displacement_map_var, bone_mtx_opt, v_weights_opt, kd_map_opt, ks_map_opt, normal_map_opt, *args, **kwargs):
        # ==============================================================================================
        #  Setup material for optimized mesh
        # ==============================================================================================

        opt_material = {
            # 'bsdf'   : ref_mesh.material['bsdf'],
            # 'bsdf'   : 'diffuse',
            'bsdf'   : self.FLAGS.optimizing_bsdf,
            'kd'     : kd_map_opt,
            'ks'     : ks_map_opt,
            'normal' : normal_map_opt
        }

        # ==============================================================================================
        #  Setup base mesh operation graph, precomputes topology etc.
        # ==============================================================================================

        # Create optimized mesh with trainable positions 
        opt_base_mesh = Mesh(v_pos_opt, self.normalized_base_mesh.t_pos_idx, material=opt_material, bone_mtx=bone_mtx_opt, v_weights=v_weights_opt, base=self.normalized_base_mesh)

        # Scale from [-1, 1] local coordinate space to match extents of the reference mesh
        opt_base_mesh = mesh.align_with_reference(opt_base_mesh, self.ref_mesh)

        # Compute smooth vertex normals
        opt_base_mesh = mesh.auto_normals(opt_base_mesh)

        # Set up tangent space
        opt_base_mesh = mesh.compute_tangents(opt_base_mesh)

        # Set up opt mesh skining
        if self.FLAGS.skinning:
            opt_base_mesh = mesh.skinning(opt_base_mesh)

        # Subdivide if we're doing displacement mapping
        if self.FLAGS.subdivision > 0:
            # Subdivide & displace optimized mesh
            subdiv_opt_mesh = mesh.subdivide(opt_base_mesh, steps=self.FLAGS.subdivision)
            opt_detail_mesh = mesh.displace(subdiv_opt_mesh, displacement_map_var, self.FLAGS.displacement, keep_connectivity=True)
        else:
            opt_detail_mesh = opt_base_mesh

        return opt_base_mesh, opt_detail_mesh

    def clamp_params(self):
        # ==============================================================================================
        #  Clamp trainables to reasonable range
        # ==============================================================================================
        self.opt_params["normal_map_opt"].clamp_(min=-1, max=1)
        self.opt_params["kd_map_opt"].clamp_(min=0, max=1)
        self.opt_params["ks_map_opt"].clamp_rgb_(minR=0, maxR=1, minG=self.FLAGS.min_roughness, maxG=1.0, minB=0.0, maxB=1.0)

    def render_image(self, glctx, rotate_y, render_ref_mesh, ref_mesh_aabb, mesh_scale):
        # Background color
        if self.FLAGS.background == 'checker':
            background = torch.tensor(util.checkerboard(self.FLAGS.display_res, 8), dtype=torch.float32, device='cuda')
        elif self.FLAGS.background == 'white':
            background = torch.ones((1, self.FLAGS.display_res, self.FLAGS.display_res, 3), dtype=torch.float32, device='cuda')
        else:
            background = None

        # Projection matrix
        proj_mtx = util.projection(x=0.4, f=1000.0)
        
        eye = np.array(self.FLAGS.camera_eye)
        eye = (util.rotate_y(rotate_y) @ [*eye, 1.])[:3]

        up  = np.array(self.FLAGS.camera_up)
        at  = np.array([0,0,0])
        a_mv =  util.lookAt(eye, at, up)
        a_mvp = np.matmul(proj_mtx, a_mv).astype(np.float32)[None, ...]
        a_lightpos = np.linalg.inv(a_mv)[None, :3, 3]
        a_campos = np.linalg.inv(a_mv)[None, :3, 3]

        params = {'mvp' : a_mvp, 'lightpos' : a_lightpos, 'campos' : a_campos, 'resolution' : [self.FLAGS.display_res, self.FLAGS.display_res], 
        # 'time' : random.randint(0, len(ref_mesh.bone_mtx))}
        'time' : 0}

        # Render images, don't need to track any gradients
        with torch.no_grad():
            # Center meshes
            _opt_detail = mesh.center_by_reference(self.opt_detail_mesh.eval(params), ref_mesh_aabb, mesh_scale)
            _opt_ref    = mesh.center_by_reference(render_ref_mesh.eval(params), ref_mesh_aabb, mesh_scale)

            # Render
            if self.FLAGS.subdivision > 0:
                _opt_base   = mesh.center_by_reference(self.opt_base_mesh.eval(params), ref_mesh_aabb, mesh_scale)
                img_base = render.render_mesh(glctx, _opt_base, a_mvp, a_campos, a_lightpos, self.FLAGS.light_power, self.FLAGS.display_res, 
                    num_layers=self.FLAGS.layers, background=background, min_roughness=self.FLAGS.min_roughness)
                img_base = util.scale_img_nhwc(img_base, [self.FLAGS.display_res, self.FLAGS.display_res])

            img_opt, triid = render.render_mesh_with_triid(glctx, _opt_detail, a_mvp, a_campos, a_lightpos, self.FLAGS.light_power, self.FLAGS.display_res, 
                num_layers=self.FLAGS.layers, background=background, min_roughness=self.FLAGS.min_roughness)

            # can't not pick reference triid
            img_ref = render.render_mesh(glctx, _opt_ref, a_mvp, a_campos, a_lightpos, self.FLAGS.light_power, self.FLAGS.display_res, 
                num_layers=1, spp=self.FLAGS.spp, background=background, min_roughness=self.FLAGS.min_roughness)

            img_triangle_errors = self.get_triangles_errors_per_pixel(triid)

            # Rescale
            img_opt  = util.scale_img_nhwc(img_opt,  [self.FLAGS.display_res, self.FLAGS.display_res])
            img_ref  = util.scale_img_nhwc(img_ref,  [self.FLAGS.display_res, self.FLAGS.display_res])
            img_error = util.scale_img_nhwc(img_triangle_errors,  [self.FLAGS.display_res, self.FLAGS.display_res])


            if self.FLAGS.subdivision > 0:
                img_disp = torch.clamp(torch.abs(self.opt_params["displacement_map_var"][None, ...]), min=0.0, max=1.0).repeat(1,1,1,3)
                img_disp = util.scale_img_nhwc(img_disp, [self.FLAGS.display_res, self.FLAGS.display_res])
                result_image = torch.cat([img_base, img_opt, img_ref], axis=2)
            else:
                result_image = torch.cat([img_opt, img_ref, img_error], axis=2)

        result_image[0] = util.tonemap_srgb(result_image[0])
        np_result_image = result_image[0].detach().cpu().numpy()
        return np_result_image


    def init_triangles_errors(self):
        print("init triangles error accumulation!!!!!")
        tri_amount = self.opt_detail_mesh.eval().t_pos_idx.size(0)
        device = self.opt_detail_mesh.eval().t_pos_idx.device
        self.triangles_errors = torch.zeros((tri_amount+1), device=device)
        self.triangles_errors_cnt = torch.ones((tri_amount+1), device=device).int()
    
    def reset_trianlge_errors(self):
        if not hasattr(self, "triangles_errors"):
            self.init_triangles_errors()
        
        self.triangles_errors[...] = 0
        self.triangles_errors_cnt[...] = 1

    @torch.no_grad()
    def update_triangles_errors(self, tri_id_perpixel, loss_per_pixel):
        """
        borrow from nerf2mesh
        """
        import torch_scatter
        if not hasattr(self, "triangles_errors"):
            self.init_triangles_errors()

        tri_id_perpixel = tri_id_perpixel.reshape(-1)
        loss_per_pixel = loss_per_pixel.mean(-1).reshape(-1)
        # include 0 as non-triangle error
        torch_scatter.scatter_add(src=loss_per_pixel, index=tri_id_perpixel, out=self.triangles_errors)
        torch_scatter.scatter_add(torch.ones_like(loss_per_pixel).int(), tri_id_perpixel, out=self.triangles_errors_cnt)

    @torch.no_grad()
    def get_triangles_errors_per_pixel(self, tri_id_perpixel):
        """
        render triangle error by tri_id_perpixel according to self.triangles_errors
        """
        
        if not hasattr(self, "triangles_errors"):
            self.init_triangles_errors()
        
        tri_amount = self.opt_detail_mesh.eval().t_pos_idx.size(0)
        device = self.opt_detail_mesh.eval().t_pos_idx.device

        appeared = torch.unique(tri_id_perpixel) # find out every single appeared triangle
        appeared = appeared[1:] # except the first one which is the background 
        faces_color = torch.zeros((tri_amount+1, 3), device=device) # initialize all faces a zero color
        faces_color[..., 0] = self.triangles_errors / self.triangles_errors_cnt
        faces_color[0] = torch.zeros(3, device=device) # background color
        triviz = faces_color[tri_id_perpixel] # sample face color of each pixel
        return triviz.mean(dim=-1, keepdim=True).repeat([1, 1, 1, 3])

    @torch.no_grad()
    def save_ply(self, path, face_quality=None):
        vertex_dt = np.dtype([('x', '<f8'), ('y', '<f8'), ('z', '<f8')])
        face_dt = np.dtype([('vertex_indices', 'O'), ('texcoord', 'O'), ('quality', '<f8')])

        # convert format
        vertices = self.opt_detail_mesh.eval().v_pos.double().cpu().numpy()
        faces = self.opt_detail_mesh.eval().t_pos_idx.cpu().numpy()
        uv = self.opt_detail_mesh.eval().v_tex.double().cpu().numpy()
        uv_idx = self.opt_detail_mesh.eval().t_tex_idx.cpu().numpy()

        vertices = [tuple(line) for line in vertices]
        faces = [np.array(line) for line in faces]

        uv = [np.array(line) for line in uv[uv_idx].reshape(-1, 6)]

        vertex_ = np.zeros(len(vertices), dtype=vertex_dt)
        face_ = np.zeros(len(faces), dtype=face_dt)


        if face_quality is None:
            face_quality = np.zeros((len(faces), ), dtype="<f8")
        else:
            assert face_quality.shape == (len(faces), )

        vertex_[:] = vertices
        face_['vertex_indices'][:] = faces
        face_['texcoord'][:] = uv
        face_['quality'][:] = face_quality

        vertex_el = PlyElement.describe(vertex_, "vertex")
        face_el = PlyElement.describe(face_, 'face', val_types={'texcoord': 'f4'})
        PlyData([vertex_el, face_el]).write(path)

    def load_ply(cls, path) -> "Mesh":
        vertex_dt = np.dtype([('x', '<f8'), ('y', '<f8'), ('z', '<f8')])
        face_dt = np.dtype([('vertex_indices', 'O'), ('texcoord', 'O'), ('quality', '<f8')])

        ply = PlyData.read(path)
        assert ply.elements[0].dtype() == vertex_dt
        assert ply.elements[1].dtype() == face_dt

        vertices = np.array([list(line) for line in ply.elements[0].data])
        faces = np.array([list(line) for line in ply.elements[1]['vertex_indices']])

        uv_ = np.array([list(line) for line in ply.elements[1]['texcoord']])
        _, index = np.unique(faces, return_index=True)
        uv = uv_.reshape(-1, 2)[index]
        normals = np.ones_like(vertices)


        vertices, faces, uv = vertices.astype(np.float32).copy(), faces.astype(np.int32).copy(), uv.astype(np.float32).copy()
        
        return Mesh(
            v_pos=torch.tensor(vertices, device="cuda", dtype=torch.float32),
            t_pos_idx=torch.tensor(faces, device="cuda", dtype=torch.int64),
            v_nrm=torch.tensor(normals, device="cuda", dtype=torch.float32),
            t_nrm_idx=torch.tensor(faces, device="cuda", dtype=torch.int64),
            v_tex=torch.tensor(uv, device="cuda", dtype=torch.float32),
            t_tex_idx=torch.tensor(faces, device="cuda", dtype=torch.int64),
        )


    @torch.no_grad()
    def decimate(self, decimate_ratio=0.1, out_dir="./", epoch=0):
        print("decimate" + "!"*5)
        import pymeshlab as pml

        errors = (self.triangles_errors / self.triangles_errors_cnt).cpu().numpy()
        errors = errors[1:] # remove background loss
        thresh_decimate = np.percentile(errors, 50)
        mask = np.zeros_like(errors)
        mask[(errors < thresh_decimate)] = 1

        self.save_ply(f"{out_dir}/decimating_{epoch}.ply", face_quality=mask)

        ms = pml.MeshSet()
        ms.load_new_mesh(f"{out_dir}/decimating_{epoch}.ply")
        ms.compute_selection_by_condition_per_face(condselect='fq == 1')
        ms.meshing_decimation_quadric_edge_collapse_with_texture(targetfacenum=int((1 - decimate_ratio) * (mask == 1).sum()), selected=True)
        ms.save_current_mesh(f"{out_dir}/decimated_{epoch}.ply", save_textures=False)

        base_mesh = self.load_ply(f"{out_dir}/decimated_{epoch}.ply")
        print("verts, faces", len(base_mesh.v_pos), len(base_mesh.t_pos_idx))
        return self.__class__(self.FLAGS, base_mesh, self.ref_mesh)

    @classmethod
    def loadmesh(cls, path, basemesh:Mesh) -> "Mesh":
        import pymeshlab as pml
        ms = pml.MeshSet()
        ms.load_new_mesh(path)
        try:
            ms.convert_perwedge_uv_into_pervertex_uv()
        except:
            print("Convert perwedge to pervertex texture uv failed!")

        m = ms.current_mesh()
        vertices, faces, normals = m.vertex_matrix(), m.face_matrix(), m.vertex_normal_matrix()
        if m.has_vertex_tex_coord():
            uv = m.vertex_tex_coord_matrix()
        else:
            raise NotImplementedError("this mesh has no uv?")
        
        return Mesh(v_pos=vertices, t_pos_idx=faces, v_nrm=normals, t_nrm_idx=faces, v_tex=uv, t_tex_idx=faces, base=basemesh)
    
    @classmethod
    def dumpmesh(cls, path, mesh:"Mesh", *args, **kwargs):
        obj.write_obj(path, mesh, save_mtl=False)

    def dump(self, out_dir, *args, **kwargs):
        obj.write_obj(os.path.join(out_dir, "mesh/"), self.opt_base_mesh.eval(), *args, **kwargs)

###############################################################################
# Main shape fitter function / optimization loop
###############################################################################

def optimize_mesh(
    FLAGS,
    out_dir, 
    log_interval=10,
    mesh_scale=2.0
    ):

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "mesh"), exist_ok=True)

    # Guess learning rate if not specified
    if FLAGS.learning_rate is None:
        FLAGS.learning_rate = 0.01

    # Reference mesh
    ref_mesh = load_mesh(FLAGS.ref_mesh, FLAGS.mtl_override)
    print("Ref mesh has %d triangles and %d vertices." % (ref_mesh.t_pos_idx.shape[0], ref_mesh.v_pos.shape[0]))

    # Check if the training texture resolution is acceptable
    ref_texture_res = np.maximum(ref_mesh.material['kd'].getRes(), ref_mesh.material['ks'].getRes())
    if 'normal' in ref_mesh.material:
        ref_texture_res = np.maximum(ref_texture_res, ref_mesh.material['normal'].getRes())
    if FLAGS.texture_res[0] < ref_texture_res[0] or FLAGS.texture_res[1] < ref_texture_res[1]:
        print("---> WARNING: Picked a texture resolution lower than the reference mesh [%d, %d] < [%d, %d]" % (FLAGS.texture_res[0], FLAGS.texture_res[1], ref_texture_res[0], ref_texture_res[1]))

    # Base mesh
    base_mesh = load_mesh(FLAGS.base_mesh)
    print("Base mesh has %d triangles and %d vertices." % (base_mesh.t_pos_idx.shape[0], base_mesh.v_pos.shape[0]))
    print("Avg edge length: %f" % regularizer.avg_edge_length(base_mesh))


    assert not FLAGS.random_train_res or FLAGS.custom_mip, "Random training resolution requires custom mip."

    # ==============================================================================================
    #  Setup reference mesh. Compute tangentspace and animate with skinning
    # ==============================================================================================

    render_ref_mesh = mesh.compute_tangents(ref_mesh)

    # set up skinning for reference mesh
    if FLAGS.skinning:
        render_ref_mesh = mesh.skinning(render_ref_mesh)
    
    # Compute AABB of reference mesh. Used for centering during rendering TODO: Use pre frame AABB?
    ref_mesh_aabb = mesh.aabb(render_ref_mesh.eval())

    trainable_mesh = TrainableMesh(FLAGS, base_mesh, ref_mesh)

    # Laplace regularizer
    if FLAGS.relative_laplacian:
        with torch.no_grad():
            orig_opt_base_mesh = trainable_mesh.opt_base_mesh.eval().clone()
        lap_loss_fn = regularizer.laplace_regularizer_const(trainable_mesh.opt_detail_mesh, orig_opt_base_mesh)
    else:
        lap_loss_fn = regularizer.laplace_regularizer_const(trainable_mesh.opt_detail_mesh)

    # ==============================================================================================
    #  Image loss
    # ==============================================================================================
    image_loss_fn = createLoss(FLAGS)

    # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    img_cnt = 0
    ang = 0.0
    img_loss_vec = []
    lap_loss_vec = []
    iter_dur_vec = []
    # glctx = dr.RasterizeGLContext()
    glctx = dr.RasterizeCudaContext()

    # Projection matrix
    proj_mtx = util.projection(x=0.4, f=1000.0)

    for it in range(FLAGS.iter+1):
        # ==============================================================================================
        #  Display / save outputs. Do it before training so we get initial meshes
        # ==============================================================================================

        # Show/save image before training step (want to get correct rendering of input)
        display_image = FLAGS.display_interval and (it % FLAGS.display_interval == 0)
        save_image = FLAGS.save_interval and (it % FLAGS.save_interval == 0)
        if display_image or save_image:
            if display_image:
                disit = it // FLAGS.display_interval
                np_result_image = trainable_mesh.render_image(glctx, disit * (np.pi / 20), render_ref_mesh, ref_mesh_aabb, mesh_scale)
                util.display_image(np_result_image, size=FLAGS.display_res,  title='%d / %d' % (it, FLAGS.iter))
            if save_image:
                saveit = it // FLAGS.save_interval
                np_result_image = trainable_mesh.render_image(glctx, saveit * (np.pi / 10), render_ref_mesh, ref_mesh_aabb, mesh_scale)
                util.save_image(out_dir + '/' + ('img_%06d.png' % img_cnt), np_result_image)
                # decimate the mesh
                trainable_mesh = trainable_mesh.decimate(out_dir=out_dir+"/decimate", epoch=img_cnt)
                img_cnt += 1

        # ==============================================================================================
        #  Initialize training
        # ==============================================================================================
        iter_start_time = time.time()
        img_loss = torch.zeros([1], dtype=torch.float32, device='cuda')
        lap_loss = torch.zeros([1], dtype=torch.float32, device='cuda')

        iter_res = FLAGS.train_res
        iter_spp = FLAGS.spp
        if FLAGS.random_train_res:
            # Random resolution, 16x16 -> train_res. Scale up sample count so we always land close to train_res*samples_per_pixel samples
            iter_res = np.random.randint(16, FLAGS.train_res+1)
            iter_spp = FLAGS.spp * (FLAGS.train_res // iter_res)

        mvp = np.zeros((FLAGS.batch, 4,4),  dtype=np.float32)
        campos   = np.zeros((FLAGS.batch, 3), dtype=np.float32)
        lightpos = np.zeros((FLAGS.batch, 3), dtype=np.float32)

        # ==============================================================================================
        #  Build transform stack for minibatching
        # ==============================================================================================
        for b in range(FLAGS.batch):
            if FLAGS.mycl:
                # rotate along y axis
                r_rot = util.rotate_y(np.random.uniform(-np.pi, np.pi))
                r_mv       = np.matmul(util.translate(0, 0, -RADIUS), r_rot)
                mvp[b]     = np.matmul(proj_mtx, r_mv).astype(np.float32)
                campos[b]  = np.linalg.inv(r_mv)[:3, 3]
                lightpos[b] = campos[b]
            else:
                # Random rotation/translation matrix for optimization.
                r_rot      = util.random_rotation_translation(0.25)
                r_mv       = np.matmul(util.translate(0, 0, -RADIUS), r_rot)
                mvp[b]     = np.matmul(proj_mtx, r_mv).astype(np.float32)
                campos[b]  = np.linalg.inv(r_mv)[:3, 3]
                lightpos[b] = util.cosine_sample(campos[b])*RADIUS


        params = {'mvp' : mvp, 'lightpos' : lightpos, 'campos' : campos, 'resolution' : [iter_res, iter_res], 'time' : 0}

        # Random bg color
        randomBgColor = torch.rand(FLAGS.batch, iter_res, iter_res, 3, dtype=torch.float32, device='cuda')

        # ==============================================================================================
        #  Evaluate all mesh ops (may change when positions are modified etc) and center/align meshes
        # ==============================================================================================
        _opt_ref  = mesh.center_by_reference(render_ref_mesh.eval(params), ref_mesh_aabb, mesh_scale)
        _opt_detail = mesh.center_by_reference(trainable_mesh.opt_detail_mesh.eval(params), ref_mesh_aabb, mesh_scale)

        # ==============================================================================================
        #  Render reference mesh
        # ==============================================================================================
        with torch.no_grad():
            color_ref = render.render_mesh(glctx, _opt_ref, mvp, campos, lightpos, FLAGS.light_power, iter_res, 
                spp=iter_spp, num_layers=1, background=randomBgColor, min_roughness=FLAGS.min_roughness)

        # ==============================================================================================
        #  Render the trainable mesh
        # ==============================================================================================
        
        # triid is triangle id according to each pixel
        color_opt, triid = render.render_mesh_with_triid(glctx, _opt_detail, mvp, campos, lightpos, FLAGS.light_power, iter_res, 
            spp=iter_spp, num_layers=FLAGS.layers, msaa=True , background=randomBgColor, 
            min_roughness=FLAGS.min_roughness)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        # Image-space loss
        img_loss_map = image_loss_fn(color_opt, color_ref)
        img_loss = torch.mean(img_loss_map)

        trainable_mesh.update_triangles_errors(triid, img_loss_map)

        # Compute laplace loss
        lap_loss = lap_loss_fn.eval(params)

        # Debug, store every training iteration
        if FLAGS.debug:
            from debug import save
            with torch.no_grad():
                triangle_errors_gray = trainable_mesh.get_triangles_errors_per_pixel(triid, )
                result_image = torch.cat([color_opt, color_ref, triangle_errors_gray], axis=2)
                save(result_image, 'train')

        # Log losses
        img_loss_vec.append(img_loss.item())
        lap_loss_vec.append(lap_loss.item())

        # Schedule for laplacian loss weight
        if it == 0:
            if FLAGS.laplacian_factor is not None:
                lap_fac = FLAGS.laplacian_factor
            else:
                ratio = 0.1 / lap_loss.item() # Hack that assumes RMSE ~= 0.1
                lap_fac = ratio * 0.25
            min_lap_fac = lap_fac * 0.02
        else:
            lap_fac = (lap_fac - min_lap_fac) * 10**(-it*0.000001) + min_lap_fac

        # Compute total aggregate loss
        total_loss = img_loss + lap_loss * lap_fac

        # ==============================================================================================
        #  Backpropagate
        # ==============================================================================================

        trainable_mesh.optimizer.zero_grad()
        total_loss.backward()
        trainable_mesh.optimizer.step()
        trainable_mesh.scheduler.step()

        trainable_mesh.clamp_params()

        iter_dur_vec.append(time.time() - iter_start_time)

        # ==============================================================================================
        #  Log & save outputs
        # ==============================================================================================

        # Print/save log.
        if log_interval and (it % log_interval == 0):
            img_loss_avg = np.mean(np.asarray(img_loss_vec[-log_interval:]))
            lap_loss_avg = np.mean(np.asarray(lap_loss_vec[-log_interval:]))
            iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))
            
            remaining_time = (FLAGS.iter-it)*iter_dur_avg
            print("iter=%5d, img_loss=%.6f, lap_loss=%.6f, lr=%.5f, time=%.1f ms, rem=%s" % 
                (it, img_loss_avg, lap_loss_avg*lap_fac, trainable_mesh.optimizer.param_groups[0]['lr'], iter_dur_avg*1000, util.time_to_text(remaining_time)))

    # Save final mesh to file
    trainable_mesh.dump(out_dir)

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='diffmodeling')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', type=int, default=512)
    parser.add_argument('-rtr', '--random-train-res', action='store_true', default=False)
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-si', '--save-interval', type=int, default=100)
    parser.add_argument('-lr', '--learning-rate', type=float, default=None)
    parser.add_argument('-lp', '--light-power', type=float, default=5.0)
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-sd', '--subdivision', type=int, default=0)
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-rt', '--random-textures', action='store_true', default=False)
    parser.add_argument('-lf', '--laplacian-factor', type=float, default=None)
    parser.add_argument('-rl', '--relative-laplacian', type=bool, default=False)
    parser.add_argument('-bg', '--background', default='checker', choices=['black', 'white', 'checker'])
    parser.add_argument('--loss', default='logl1', choices=['logl1', 'logl2', 'mse', 'smape', 'relativel2'])
    parser.add_argument('-o', '--out-dir', type=str, default=None)
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-rm', '--ref_mesh', type=str)
    parser.add_argument('-bm', '--base-mesh', type=str)
    parser.add_argument('--optimizing-bsdf', type=str, default="pbr")
    parser.add_argument('--only-optimize', type=str, default="", help="specify the only optimizing part of rendering parameter")
    parser.add_argument('--skinning', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--mycl', action='store_true', default=False, help="whether using my positional pattern of camera and light")

    FLAGS = parser.parse_args()

    FLAGS.camera_eye = [0.0, 0.0, RADIUS]
    FLAGS.camera_up  = [0.0, 1.0, 0.0]

    if FLAGS.only_optimize:
        # only optimize diffuse map
        FLAGS.skip_train = ['position', 'normal', 'kd', 'ks', 'displacement']
        FLAGS.skip_train.remove(FLAGS.only_optimize)
    else:
        FLAGS.skip_train = []
    
    print("skip_train:", FLAGS.skip_train)
    
    FLAGS.displacement = 0.15
    FLAGS.mtl_override = None

    if FLAGS.config is not None:
        with open(FLAGS.config) as f:
            data = json.load(f)
            for key in data:
                print(key, data[key])
                FLAGS.__dict__[key] = data[key]

    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res
    if FLAGS.out_dir is None:
        out_dir = 'out/cube_%d' % (FLAGS.train_res)
    else:
        out_dir = 'out/' + FLAGS.out_dir

    optimize_mesh(FLAGS, out_dir)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
