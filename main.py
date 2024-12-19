import argparse
import json
import os
import random

import numpy as np
import torch

from utils.organizer import Organizer
from utils.sd import StableDiffusion
from utils.preprocess_personalization import process_combinations


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default="a photo of <asset0> next to <asset1>", help="text prompt")
    parser.add_argument('--objects', default=['sofa', 'chair'], type=str, nargs="+",
                        help="which objects to use")
    parser.add_argument('--objects_idx', default=[0, 0], type=int, nargs="+",
                        help="which of the options of the objects to use")
    parser.add_argument('--camera_dist', default=6.0, type=float, help="the camera distance from the object on render")
    parser.add_argument('--relationship', default='next to', help="relationship between objects")
    parser.add_argument('--objects_root', default='objects/', help="objects location")
    parser.add_argument('--num_outputs', default=30, type=int, help="how many results to create")
    parser.add_argument('--objaverse_dir', default="objaverse/",
                        help="root path for meshes and textures")
    parser.add_argument('--optional_matches', default=5, type=int, help="how many results to create")

    parser.add_argument('--save_root', default="outputs/", help="images save path")
    parser.add_argument('--scene_description', default="a living room", help="scene description text")
    parser.add_argument('--expr_name', default="dining_table", help="text prompt")

    parser.add_argument('--extract_renders', action='store_true', help="extract initial renders using blender")

    parser.add_argument('--blender_root', default="inputs/", help="images save path")
    parser.add_argument('--combinations_path', default='inputs/combinations',
                        help="combinations save path")
    parser.add_argument('--threads', type=int, default=20)
    parser.add_argument('--blender_executable', default="blender-4.0.1-linux-x64/blender",
                        help="images save path")
    parser.add_argument('--transform_optimization', action='store_true', help="project objects to the ground")
    parser.add_argument('--common_plane_lambda', type=float, default=1e6)
    parser.add_argument('--bboxes_lambda', type=float, default=1e6)
    parser.add_argument('--neglection_threshold', type=float, default=0.39)

    parser.add_argument('--save_results', action='store_true', help="fast 3d parameters")
    parser.add_argument('--load_personalization', action='store_true', help="personalization - Text2Scene")

    parser.add_argument('--seed', type=int, default=0)

    opt = parser.parse_args()

    # Seed
    random.seed(opt.seed)
    os.environ['PYTHONHASHSEED'] = str(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.expr_name = '_'.join(opt.objects)
    object_and_idx = [f'--object {o} {i}' for o, i in zip(opt.objects, opt.objects_idx)]

    # extract renders
    if opt.extract_renders and not opt.load_personalization:
        object_string = ' '.join(object_and_idx)
        os.system(
            f'{opt.blender_executable} -P utils/blender_engine.py -b -t {opt.threads}' + \
            f' --training {object_string} --camera_dist {opt.camera_dist}' + \
            f' --objaverse {opt.objaverse_dir}'
        )
        os.system(
            f'{opt.blender_executable} -P utils/blender_engine.py -b -t {opt.threads}' + \
            f' --SI-PnP {object_string} --camera_dist {opt.camera_dist}' + \
            f' --objaverse {opt.objaverse_dir}'
        )
        print(f'[INFO] extracted renders!')


    # concept list
    text = opt.scene_description + ' with '
    concepts_list = {}
    for i, (obj, idx) in enumerate(zip(opt.objects, opt.objects_idx)):
        curr_obj = os.path.join(opt.objects_root, obj + '.json')
        with open(curr_obj, 'r') as f:
            concept = json.load(f)
        concept.update({'name': obj})
        concept['path'] = concept['path'][idx]
        concept = {f'asset{i}': concept}
        text += f'<asset{i}> next to '
        concepts_list.update(concept)
    opt.text = text[:-9]

    output_name = ''.join([os.path.basename(v['path'])[:4] for v in concepts_list.values()])
    objects_hash = [os.path.basename(v['path']).replace('.glb', '') for v in concepts_list.values()]
    opt.save_path = os.path.join(opt.save_root, opt.expr_name, output_name)

    list_of_objects_dirs = [os.path.join(opt.blender_root, 'personalization', o, h) for o, h in zip(opt.objects, objects_hash)]
    combined_imgs_and_masks_dir = "__".join([f'{o}_{i}' for o, i in zip(opt.objects, objects_hash)])
    combinations_dir = os.path.join(opt.combinations_path, combined_imgs_and_masks_dir)
    process_combinations(list_of_objects_dirs,
                         output_dir_path=combinations_dir)

    # guidance
    guidance = StableDiffusion(device, opt, concepts_list, combinations_dir=combinations_dir)
    organizer = Organizer(opt, guidance, concepts_list, object_and_idx, device=device)
    organizer.run()
