import random

import numpy as np
import torch
import torch.utils.data


class GraspDatasetBase(torch.utils.data.Dataset):
    """
    An abstract dataset for training networks in a common format.
    """

    def __init__(self, output_size=224, include_depth=False, include_rgb=True, random_rotate=True,
                 random_zoom=False, input_only=False):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        """
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_rgb = include_rgb

        self.grasp_files = []

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if(idx >= len(self.grasp_files)):
            idx = idx % len(self.grasp_files)

        if self.random_rotate:
            rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0


        bbs = self.get_gtbb(idx, rot, zoom_factor)
        top, left, bottom, right = bbs.box_that_fits_all_graps

        if top > 0 and bottom <= 224:
            if top >= 224 - bottom:
                shift_y = -random.randint(0, ( 224 - bottom) )
            else:
                shift_y =  random.randint(0, top)
        else:
            shift_y = 0

        if left > 0 and right <= 224:
            if left >= 224 - right:
                shift_x =  -random.randint(0, left)
            else:
                shift_x = random.randint(0, ( 224 - right))
        else:
            shift_x = 0


        # Load the depth image
        if self.include_depth:
            depth_img_obj = self.get_depth(idx, rot, zoom_factor)
            depth_img_obj.offset_img(  shift_x, shift_y )
            depth_img = depth_img_obj.img
        # Load the RGB image
        if self.include_rgb:
            rgb_img_obj = self.get_rgb(idx, rot, zoom_factor)

        # Load the grasps
        #print(idx)



        #print( idx )
        shift_y = 0

        bbs.offset( (shift_y, shift_x) )
        rgb_img_obj.offset_img( shift_x, shift_y )

        rgb_img = rgb_img_obj.img[:, :, :]
        
        #print( rgb_img.shape )

        """
        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        width_img = np.clip(width_img, 0.0, self.output_size / 2) / (self.output_size / 2)
        """
        #print("gets called")
        target = bbs.draw_detr()
        target['image_id'] = torch.tensor([idx])
        target['rot'] = torch.tensor([rot])
        target['zoom'] = torch.tensor([zoom_factor])
        target['shift_x'] = torch.tensor([shift_x])
        target['shift_y'] = torch.tensor([shift_y])
        
        
        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )

        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        """
        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2 * ang_img))
        sin = self.numpy_to_torch(np.sin(2 * ang_img))
        width = self.numpy_to_torch(width_img)
        """

        #print(x.shape)
        #print(target)
        return x, target

    def __len__(self):
        return len(self.grasp_files) * 10
