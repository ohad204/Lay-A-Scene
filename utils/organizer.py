import os
import base64
from copy import copy

import cv2
import numpy as np
import imageio.v2 as imageio
import torch
import torch.nn as nn

from torchvision.transforms import PILToTensor

from utils.SIPnP import SIPnP


def find_matches(keypoints_dict, target_ft, n_matches=2):
    # Convert keypoints_dict into separate lists for keypoints and descriptors
    keypoints_list = list(keypoints_dict.keys())
    descriptors_list = [keypoints_dict[keypoint].cpu().squeeze().numpy() for keypoint in keypoints_list]
    descriptors_list = np.stack(descriptors_list, axis=1).T
    target_ft_upsampled = target_ft.cpu().numpy()

    # Extract descriptors for the keypoints from the target_ft
    grid_x, grid_y = torch.meshgrid(torch.arange(32), torch.arange(32), indexing='ij')
    img_grid = torch.stack((grid_x, grid_y), dim=2)
    keypoints_list_tr = img_grid.reshape(-1, 2)
    target_descriptors = [target_ft_upsampled[:, y, x] for (x, y) in keypoints_list_tr]
    target_descriptors = np.stack(target_descriptors, axis=1)

    # Find matching pairs - cross-checking
    cosine = descriptors_list @ target_descriptors
    sorted_indices = np.argsort(-cosine, axis=1)
    top2_indices = sorted_indices[:, :n_matches]
    matches_ = [(i, o) for i, p in enumerate(top2_indices) for o in p]
    sorted_max = np.sort(-cosine, axis=1)
    matches_max_ = - sorted_max[:, :n_matches].reshape(-1, 1)

    matches = []
    matches_max = []
    for p, m in zip(matches_, matches_max_):
        argmax_fit = cosine[:, p[1]].argmax()
        if p[0] == argmax_fit:
            matches.append(p)
            matches_max.append(m)

    matches_dict = {}
    for match in matches:
        src_point = keypoints_list[match[0]]
        tgt_point = keypoints_list_tr[match[1]]  # Using keypoints_list for target as we're looking at same locations
        matches_dict[src_point] = tgt_point

    return matches_dict, np.array(matches_max)


class Organizer(object):
    def __init__(self,
                 opt,  # extra conf
                 guidance,  # guidance network
                 concepts_list,  # concept list
                 object_and_idx,  # object and idx
                 local_rank=0,  # which GPU am I
                 device=None,  # device to use, usually setting to None is OK. (auto choose device)
                 ):
        self.opt = opt
        self.concepts_list = concepts_list
        output_name = ''.join([os.path.basename(v['path'])[:4] for v in self.concepts_list.values()])
        self.save_path = os.path.join(opt.save_root, opt.expr_name, output_name)
        self.expr_name = opt.expr_name
        self.object_and_idx = object_and_idx
        self.num_outputs = opt.num_outputs
        self.optional_matches = self.opt.optional_matches
        self.device = device if device is not None else torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.background = imageio.imread('utils/gray_blue_512_512.png')

        # assets information
        self.assets_info = [{'asset': f'<{k}>', 'name': v['name'], 'class_prompt': v['class_prompt'], 'surface': None,
                             'path': os.path.basename(v['path']).replace('.glb', '')} for k, v in
                            self.concepts_list.items()]
        self.p3p_features = {'<' + k + '>': {} for k in self.concepts_list.keys()}

        self.K = None
        self.img_to_tensor = PILToTensor()

        # blender
        self.blender_root = opt.blender_root.replace('personalization', 'SI-PnP')

        # guide model
        self.guidance = guidance

        self.dift_ratio = 16
        self.neglection_threshold = self.opt.neglection_threshold


    @staticmethod
    def normalize_image(img):
        """Normalize an image to be in the range [0, 1]."""
        return (img - img.min()) / (img.max() - img.min())

    @staticmethod
    def get_3d_points(keypoints, depth, K, M):
        ys = np.floor(keypoints[:, 1]).astype(np.int32)
        xs = np.floor(keypoints[:, 0]).astype(np.int32)

        z = depth[ys, xs]

        K_inv = np.linalg.inv(K)
        point_cloud = (K_inv @ np.stack((xs, ys, np.ones_like(xs)))) * z[np.newaxis, :]
        M_inv = np.linalg.inv(M)
        point_cloud = M_inv[:3, :3] @ point_cloud + M_inv[:3, 3:]
        return point_cloud.T

    @staticmethod
    def encode_matrix_to_base64(matrix):
        binary_string = matrix.tobytes()
        base64_string = base64.b64encode(binary_string).decode('utf-8')
        return base64_string

    def process_object(self, sd_ft, prompt, obj):

        obj_name = obj['name']
        idx_name = obj['path']
        parameters = {'name': obj_name, 'score': 0.0}

        render_cameras = np.load(f'{self.blender_root}/SI-PnP/{obj_name}/{idx_name}/cameras.npz')
        scale = render_cameras['scale']
        floor = render_cameras['floor']

        sd_ft, sd_kp = sd_ft

        # for each object render at several different object scales
        points_3d = []
        points_2d = []
        points_score = []
        for azimuth in range(-50, 60, 10):

            K = render_cameras['K_%03d_%03d' % (azimuth, 30)]
            M = render_cameras['M_%03d_%03d' % (azimuth, 30)]

            if self.K is None:
                self.K = K

            if azimuth in self.p3p_features[prompt].keys():
                mesh_depth = self.p3p_features[prompt][azimuth]['depth']
                rendered_image = self.p3p_features[prompt][azimuth]['render']
                mesh_alpha = rendered_image[:, :, -1].unsqueeze(0).unsqueeze(0)
                mesh_img = rendered_image[:, :, :3]
                mesh_img = mesh_img.permute(2, 0, 1).unsqueeze(0).float()
                mesh_img = (mesh_img - 0.5) * 2
                render_ft = self.p3p_features[prompt][azimuth]['ft']

            else:
                mesh_depth = cv2.imread(
                    f'{self.blender_root}/SI-PnP/{obj_name}/{idx_name}/depth_{azimuth}_{30}_depth.png0001.exr',
                    cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                mesh_depth = mesh_depth[:, :, 0]

                rendered_image = imageio.imread(f'{self.blender_root}/SI-PnP/{obj_name}/{idx_name}/img_{azimuth}_30_.png')
                rendered_image[rendered_image[..., 3] < 50, :3] = self.background[rendered_image[..., 3] < 50]

                rendered_image = torch.from_numpy(rendered_image).to(self.device).float() / 255.0
                mesh_alpha = rendered_image[:, :, -1].unsqueeze(0).unsqueeze(0)
                mesh_img = rendered_image[:, :, :3]

                mesh_img = mesh_img.permute(2, 0, 1).unsqueeze(0).float()
                mesh_img = (mesh_img - 0.5) * 2

                # find render image features
                render_ft = self.guidance.pipe.dift(mesh_img, prompt=prompt)

                self.p3p_features[prompt].update(
                    {azimuth: {'depth': mesh_depth, 'render': rendered_image, 'ft': render_ft}})

            # take as keypoints only pixels with the object
            mesh_alpha = nn.functional.interpolate(mesh_alpha, size=32).squeeze(0).squeeze(0)
            keypoints = torch.where(mesh_alpha > 0.5)
            keypoints = list(zip(keypoints[0], keypoints[1]))
            keypoint_feature_dict = {(int(v), int(u)): render_ft[:, u, v] for u, v in keypoints}

            # find matches between sd image and the object rendering
            matches_dict, matches_score = find_matches(keypoints_dict=keypoint_feature_dict, target_ft=sd_ft, n_matches=self.optional_matches)

            # extract 3d points from the matches
            object_pts = np.array(list(matches_dict.keys()))
            object_pts *= self.dift_ratio
            sd_pts = np.array([np.array(x) for x in list(matches_dict.values())])

            # get 3d points
            points_3d.append(self.get_3d_points(object_pts, mesh_depth, K, M))
            points_2d.append(sd_pts.astype(np.double) * self.dift_ratio)
            points_score.append(matches_score)

        # --------------------------- aggregate points and find initialization for SI-PnP ------------------------------

        if len(points_3d) == 0:
            return parameters, 0, 0

        points_3d = np.concatenate(points_3d, axis=0) * scale
        points_2d = np.concatenate(points_2d, axis=0)
        points_score = np.concatenate(points_score, axis=0)

        if points_3d.shape[0] < 4:
            return parameters, 0, 0

        success, rotation_vector, translation_vector, curr_inliners = cv2.solvePnPRansac(points_3d,
                                                                                         points_2d,
                                                                                         K,
                                                                                         np.zeros(4),
                                                                                         flags=cv2.SOLVEPNP_P3P,
                                                                                         iterationsCount=1000,
                                                                                         )

        inliners_points_3d = points_3d[curr_inliners[..., 0]]
        inliners_points_2d = points_2d[curr_inliners[..., 0]]
        inliners_points_score = points_score[curr_inliners[..., 0]]

        match_score = np.median(inliners_points_score)

        parameters['inliners'] = (inliners_points_3d, inliners_points_2d)
        parameters['matching_score'] = match_score
        parameters['rotation_vector'] = rotation_vector
        parameters['translation_vector'] = translation_vector

        return parameters, scale, floor


    def find_3d_arrangment(self, frame=0):

        # create target image
        sd_image = self.guidance.pipe(self.opt.text)
        sd_image = sd_image.images[0]

        # normalize image for dift features
        img_tensor = (self.img_to_tensor(sd_image) / 255.0 - 0.5) * 2
        img_tensor = img_tensor.to(self.device)

        # initialize parameters dict
        parameters = {f'<{obj}>': {} for obj in self.concepts_list.keys()}
        parameters.update({'sd_image': np.array(sd_image)})

        object_and_idx = copy(self.object_and_idx)
        matching_score = []
        for i, obj in enumerate(self.assets_info):
            asset = obj["asset"]
            prompt = obj["asset"]

            sd_ft = self.guidance.pipe.dift(img_tensor, prompt=prompt, up_ft_index=1)
            sd_kp = None

            object_parameter, scale, floor = self.process_object((sd_ft, sd_kp), prompt, obj)
            parameters[asset] = object_parameter
            obj['surface'] = floor
            obj['rotations_vectors'] = object_parameter['rotation_vector']
            obj['translation_vector'] = object_parameter['translation_vector']

            if object_parameter['matching_score'] < self.neglection_threshold:
                print(f'skipping {frame}')
                return None, None, None

            matching_score.append(object_parameter['matching_score'])
            object_and_idx[i] += f' --scale {scale}'

        # common plane finder
        inliners = [v['inliners'] for k, v in parameters.items() if 'asset' in k]
        surface_points = [o['surface'] for o in self.assets_info]
        translations = [o['translation_vector'] for o in self.assets_info]
        rotations = [o['rotations_vectors'] for o in self.assets_info]

        CommonSurface = SIPnP(surface_points, self.K, inliners, rotations, translations,
                                         common_plane_lambda=self.opt.common_plane_lambda,
                                         device=self.device)
        optimized_rotations, optimized_translations, floor_projection = CommonSurface.get_transforms()

        for i, (r, t) in enumerate(zip(optimized_rotations, optimized_translations)):
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = r
            transform_matrix[:3, 3] = t[..., 0]
            if self.opt.transform_optimization:
                transform_matrix = transform_matrix @ floor_projection
            base64_transform_matrix = self.encode_matrix_to_base64(transform_matrix)
            object_and_idx[i] += f' --transform {base64_transform_matrix}'

        object_string = ' '.join(object_and_idx)
        blender_cmd = f'{self.opt.blender_executable} -P utils/blender_engine.py -b -t {self.opt.threads}' + \
            f' --evaluate {object_string} --output {self.save_path} --camera_dist {self.opt.camera_dist}' + \
            f' --objaverse {self.opt.objaverse_dir}'
        os.system(blender_cmd)

        # save  parameters
        if self.opt.save_results:
            parameters_path = os.path.join(self.save_path, f'parameters_{frame}.npy')
            np.save(parameters_path, parameters, allow_pickle=True)

        imgs_gif = []
        anglez = 10
        for anglex in range(-180, 181, 5):
            img = imageio.imread(f"{self.save_path}/%d_%d____.png" % (anglex, anglez))
            img[img[..., 3] < 50, :3] = self.background[img[..., 3] < 50]
            concat_pred_img = img[..., :3]
            imgs_gif.append(concat_pred_img)

        return imgs_gif, np.array(sd_image), matching_score


    def run(self):
        print(f'[INFO] extracting objecs positions')
        os.makedirs(os.path.join(self.save_path, self.expr_name), exist_ok=True)

        # try several times
        for i in range(self.num_outputs):

            pred_rgbs, sd_image, natching_score = self.find_3d_arrangment(i)
            if pred_rgbs is None:
                continue
            else:
                imageio.mimsave(os.path.join(self.save_path, f'result_video.gif'), pred_rgbs,
                                duration=0.01, loop=10000)
                break

        print('finished')