import glob
import os

from utils.dataset_processing import grasp, image
from .grasp_data import GraspDatasetBase


class CornellDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Cornell dataset.
    """

    def __init__(self, file_path, ds_rotate=0, duplication_factor = 3, **kwargs):
        """
        :param file_path: Cornell Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(CornellDataset, self).__init__(**kwargs)

        self.grasp_files = glob.glob(os.path.join(file_path, '*', 'pcd*cpos.txt'))
        self.grasp_files.sort()
        self.length = len(self.grasp_files) * duplication_factor
        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]

        self.depth_files = [f.replace('cpos.txt', 'd.tiff') for f in self.grasp_files]
        self.rgb_files = [f.replace('d.tiff', 'r.png') for f in self.depth_files]

    # This function helps with generating an augmented dataset if rot and zoom are set
    # to something else than the default parameters
    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top

    # Load the grasp rectangle
    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        gtbbs.rotate(rot, center)
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs

    # This function helps with generating an augmented dataset if rot and zoom are set
    # to something else than the default parameters, also sets the size the max size to (480, 640)
    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img


    # This function helps with generating an augmented dataset if rot and zoom are set
    # to something else than the default parameters, also sets the max size to (480, 640)
    #drop the blue channel to make it 3 channels with the depth
    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))

        # to drop the blue channel rgb_img.img[:2, :, :]

        return rgb_img

    # For visualizing data
    def get_rgb_img(self, idx,  rot=0, zoom=1.0):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        #rgb_img.normalise()
        rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img

    # get depth image for visulaization
    def get_depth_img(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        depth_img.inpaint()

        return depth_img