import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid
# from scene.grid import HashHexPlane
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        # breakpoint()
        self.args = args
        # self.args.empty_voxel=True
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1),nn.Linear(self.W, 1))
        
        self.ratio=0
        self.create_net()
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe !=0:
            
            grid_out_dim = self.grid.feat_dim+(self.grid.feat_dim)*2 
        else:
            grid_out_dim = self.grid.feat_dim
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim ,self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        mask_deform = []
        test_len = 2
        for i in range(test_len):
            mask_deform.append(nn.ReLU())
            mask_deform.append(nn.Linear(self.W,self.W))
        mask_deform.append(nn.ReLU())
        mask_deform.append(nn.Linear(self.W, 1))
        self.mask_deform = nn.Sequential(*mask_deform)

        light_deform = []
        for i in range(test_len):
            light_deform.append(nn.ReLU())
            light_deform.append(nn.Linear(self.W,self.W))
        light_deform.append(nn.ReLU())
        light_deform.append(nn.Linear(self.W, 1))
        self.light_deform = nn.Sequential(*light_deform)

        light_deform_dy = []
        for i in range(test_len):
            light_deform_dy.append(nn.ReLU())
            light_deform_dy.append(nn.Linear(self.W, self.W))
        light_deform_dy.append(nn.ReLU())
        light_deform_dy.append(nn.Linear(self.W, 1))
        self.light_deform_dy = nn.Sequential(*light_deform_dy)

        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))
        # self.mask_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        # self.mask_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU()
        #                                  ,nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        # self.light_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        # self.mask_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU()
        #                                  ,nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        # self.light_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU()
        #                                  ,nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb):

        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:

            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            # breakpoint()
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature,self.grid_pe)
            hidden = torch.cat([grid_feature],-1) 
        
        
        hidden = self.feature_out(hidden)   
 

        return hidden
    @property
    def get_empty_ratio(self):
        return self.ratio
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None, time_feature=None, time_emb=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb):
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb)
        short_cut = True
        if self.args.static_mlp:
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:,:3])
        else:
            mask = torch.ones_like(opacity_emb[:,0]).unsqueeze(-1)
        # breakpoint()
        if self.args.no_dx:
            pts = rays_pts_emb[:,:3]
        else:
            dx = self.pos_deform(hidden)
            pts = torch.zeros_like(rays_pts_emb[:,:3])
            if short_cut:
                pts = rays_pts_emb[:,:3]*mask + dx
            else:
                pts = dx
        if self.args.no_ds :
            
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)

            scales = torch.zeros_like(scales_emb[:,:3])
            if short_cut:
                scales = scales_emb[:,:3]*mask + ds
            else:
                scales = ds

        if self.args.no_dr :
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)

            rotations = torch.zeros_like(rotations_emb[:,:4])
            if short_cut:
                if self.args.apply_rotation:
                    rotations = batch_quaternion_multiply(rotations_emb, dr)
                else:
                    rotations = rotations_emb[:,:4] + dr
            else:
                rotations = dr


        if self.args.no_do :
            opacity = opacity_emb[:,:1] 
        else:
            do = self.opacity_deform(hidden) 
          
            opacity = torch.zeros_like(opacity_emb[:,:1])
            if short_cut:

                opacity = opacity_emb[:,:1]*mask + do
            else:
                opacity = do
        if self.args.no_dshs:

            if not self.args.no_dx:

                #### mask
                # shs_temp = shs_emb[:, :16, :]
                # d_mask = self.mask_deform(hidden).reshape([shs_emb.shape[0], 9, 3])
                # # mask_temp = shs_emb[:, 16:, :] * mask.unsqueeze(-1) + d_mask
                # mask_temp =  d_mask
                # shs = torch.cat([shs_temp, mask_temp], 1)

                ## new mask
                shs_temp = shs_emb
                mask_deform = self.mask_deform(hidden)
                light_deform = self.light_deform(hidden)
                light_deform_dy = self.light_deform_dy(hidden)
                # shs_temp[:,-1, 2] = mask_deform[:,-1]
                # shs_temp[:, -1, 1] = light_deform[:,-1]
                # shs_temp[:, -1, 0] = light_deform_dy[:,-1]
                shs_temp[:,-1, 2] +=  mask_deform[:,-1]
                shs_temp[:, -1, 1] += light_deform[:,-1]
                shs_temp[:, -1, 0] += light_deform_dy[:,-1]

                # shs_temp[:, -1, 1:] = mask_deform
                shs = shs_temp
            else:
                shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])

            # shs = torch.zeros_like(shs_emb)
            # breakpoint()
            shs_temp = shs_emb[:,:16,:]* mask.unsqueeze(-1) + dshs
##################### mask
            # d_mask = self.mask_deform(hidden).reshape([shs_emb.shape[0],9,3])
            # mask_temp = d_mask
            # # mask_temp = shs_emb[:,16:,:]*mask.unsqueeze(-1) + d_mask
            # shs = torch.cat([shs_temp, mask_temp], 1)
            ########### na
            shs = torch.cat([shs_temp, shs_emb[:,16:,:]], 1)
            # mask_deform = self.mask_deform(hidden)
            # shs[:,-1,-1] = mask_deform[:,-1]
            # shs[:, -1, 1:] = mask_deform
            # shs_temp = shs_emb
            mask_deform = self.mask_deform(hidden)
            light_deform = self.light_deform(hidden)
            light_deform_dy = self.light_deform_dy(hidden)
            shs[:, -1, 2] = shs_emb[:, -1, 2] + mask_deform[:, -1]
            shs[:, -1, 1] = shs_emb[:, -1, 1] + light_deform[:, -1]
            shs[:, -1, 0] = shs_emb[:, -1, 0] + light_deform_dy[:, -1]
            # shs[:, -1, 2] =   mask_deform[:, -1]
            # shs[:, -1, 1] =  light_deform[:, -1]
            # shs[:, -1, 0] =  light_deform_dy[:, -1]







        return pts, scales, rotations, opacity, shs
    # def get_mlp_parameters(self):
    #     parameter_list = []
    #     for name, param in self.named_parameters():
    #         if  "grid" not in name:
    #             parameter_list.append(param)
    #     return parameter_list
    def get_mlp_parameters_cano(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name and 'mask_deform' not in name and 'light_deform' not in name and 'light_deform_dy' not in name:
                parameter_list.append(param)
        return parameter_list
    def get_mlp_parameters_others(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if 'mask_deform' not in name and 'light_deform' not in name and 'light_deform_dy' not in name:
                continue
            if "grid" not in name:
                parameter_list.append(param)
        return parameter_list

    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel)
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        # times_emb = poc_fre(times_sel, self.time_poc)
        point_emb = poc_fre(point,self.pos_poc)
        scales_emb = poc_fre(scales,self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations,self.rotation_scaling_poc)
        # time_emb = poc_fre(times_sel, self.time_poc)
        # times_feature = self.timenet(time_emb)
        means3D, scales, rotations, opacity, shs = self.deformation_net( point_emb,
                                                  scales_emb,
                                                rotations_emb,
                                                opacity,
                                                shs,
                                                None,
                                                times_sel)
        return means3D, scales, rotations, opacity, shs
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    def get_mlp_parameters_cano(self):
        return self.deformation_net.get_mlp_parameters_cano() + list(self.timenet.parameters())
    def get_mlp_parameters_others(self):
        return self.deformation_net.get_mlp_parameters_others()
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()


class deform_network_withmask(nn.Module):
    def __init__(self, args):
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth = args.defor_depth
        posbase_pe = args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2 * timebase_pe + 1
        self.timenet = nn.Sequential(
            nn.Linear(times_ch, timenet_width), nn.ReLU(),
            nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3) + (3 * (posbase_pe)) * 2,
                                           grid_pe=grid_pe, input_ch_time=timenet_output, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2 ** i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2 ** i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2 ** i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2 ** i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel)

    @property
    def get_aabb(self):
        return self.deformation_net.get_aabb

    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio

    def forward_static(self, points):
        points = self.deformation_net(points)
        return points

    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        # times_emb = poc_fre(times_sel, self.time_poc)
        point_emb = poc_fre(point, self.pos_poc)
        scales_emb = poc_fre(scales, self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations, self.rotation_scaling_poc)
        # time_emb = poc_fre(times_sel, self.time_poc)
        # times_feature = self.timenet(time_emb)
        means3D, scales, rotations, opacity, shs = self.deformation_net(point_emb,
                                                                        scales_emb,
                                                                        rotations_emb,
                                                                        opacity,
                                                                        shs,
                                                                        None,
                                                                        times_sel)
        return means3D, scales, rotations, opacity, shs

    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters()

    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb