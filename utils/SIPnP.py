import torch
from pytorch3d import transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2

class SIPnP(nn.Module):
    def __init__(self, surface, K, points, rotations, translations,
                 common_plane_lambda=10, device='cpu'):
        super(SIPnP, self).__init__()

        L = self.find_common_plane(rotations, translations, surface)

        self.device = device
        self.surface = [torch.tensor(s).to(self.device) for s in surface]
        self.L = torch.tensor(L).to(self.device)

        self.K = torch.tensor(K).double().to(self.device)
        self.points = [[torch.tensor(p).to(self.device) for p in ps] for ps in points]
        self.rotations = [torch.tensor(r).T.to(self.device) for r in rotations]
        self.translations = [torch.tensor(t).to(self.device) for t in translations]

        self.r_diff = nn.ParameterList([nn.Parameter(0.001 * torch.randn(1, 3)).to(self.device) for _ in rotations])
        self.t_diff = nn.ParameterList([nn.Parameter(0.001 * torch.randn(3, 1)).to(self.device) for _ in translations])

        self.common_plane_lambda = common_plane_lambda
        self.mse = nn.MSELoss()

        self.optimize_translations()

    def find_common_plane(self, rotations, translations, surface):
        # Transform surface to position
        rotations_matrix = [cv2.Rodrigues(r)[0] for r in rotations]
        surface_translated = [r @ p.T + t for p, r, t in zip(surface, rotations_matrix, translations)]

        # find common plane
        surface_translated = np.concatenate(surface_translated, 1).T
        centroid = np.mean(surface_translated, axis=0)
        centered_points = surface_translated - centroid
        _, _, Vt = np.linalg.svd(centered_points)
        normal_vector = Vt[-1]
        D = -np.dot(normal_vector, centroid)
        L = np.concatenate((normal_vector.reshape(-1, 1), D.reshape(-1, 1)))
        return L

    def transform_orientation(self):
        original_normal = self.L[:3, 0].cpu().numpy()
        if original_normal[1] < 0:
            desired_normal = [0, -1, 0]
        else:
            desired_normal = [0, 1, 0]

        original_normal = original_normal / np.linalg.norm(original_normal)
        desired_normal = desired_normal / np.linalg.norm(desired_normal)

        # Calculate rotation axis and angle
        rotation_axis = np.cross(original_normal, desired_normal)
        rotation_angle = np.arccos(np.dot(original_normal, desired_normal))

        rotation_matrix = self.rotation_matrix_from_axis_angle(rotation_axis, rotation_angle)
        tranlation_matrix = np.eye(4)
        tranlation_matrix[:3, :3] = rotation_matrix
        return tranlation_matrix

    @staticmethod
    def rotation_matrix_from_axis_angle(axis, angle):
        axis = axis / np.linalg.norm(axis)

        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c

        rotation_matrix = np.array(
            [[t * axis[0] ** 2 + c, t * axis[0] * axis[1] - s * axis[2], t * axis[0] * axis[2] + s * axis[1]],
             [t * axis[0] * axis[1] + s * axis[2], t * axis[1] ** 2 + c, t * axis[1] * axis[2] - s * axis[0]],
             [t * axis[0] * axis[2] - s * axis[1], t * axis[1] * axis[2] + s * axis[0], t * axis[2] ** 2 + c]])
        return rotation_matrix

    def get_rotation_translation(self):
        rot = []
        trans = []
        for r, r_d in zip(self.rotations, self.r_diff):
            current_r = r + r_d
            rot_matrix = transforms.axis_angle_to_matrix(current_r)[0].detach().cpu().numpy()
            rot.append(rot_matrix)
        for t, t_d in zip(self.translations, self.t_diff):
            current_t = t + t_d
            trans.append(current_t.detach().cpu().numpy())
        return rot, trans

    def forward(self):

        rot = [transforms.axis_angle_to_matrix(r + r_d) for r, r_d in zip(self.rotations, self.r_diff)]
        trans = [t + t_d for t, t_d in zip(self.translations, self.t_diff)]

        # Transform points
        P_transformed = [torch.matmul(r[0], p[0].T) + t for p, r, t in zip(self.points, rot, trans)]
        surface_translated = [torch.matmul(r[0], p.T) + t for p, r, t in zip(self.surface, rot, trans)]

        # Project transformed points to image plane
        P_projected = [torch.matmul(self.K, p) for p in P_transformed]
        P_projected = [p[:2, :] / p[2, :] for p in P_projected]

        # Calculate reprojection loss
        loss_matching = sum([self.mse(p.T, uv[1]) for p, uv in zip(P_projected, self.points)])

        # Add constraint that transformed points lie on the common surface
        plane = self.L
        plane_distance = [torch.matmul(plane[:3].T, p) + plane[-1] for p in surface_translated]
        loss_plane = self.common_plane_lambda * sum([torch.abs(d).sum() for d in plane_distance])

        # calculate surface loss
        loss = loss_matching + loss_plane

        return loss

    def optimize_translations(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for _ in range(5000):
            optimizer.zero_grad()
            loss = self.forward()
            loss.backward()
            optimizer.step()

    def get_transforms(self):
        rotations, translations = self.get_rotation_translation()
        floor_projection = self.transform_orientation()
        return rotations, translations, floor_projection