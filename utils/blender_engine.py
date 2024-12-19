import bpy
import os
import sys
from contextlib import contextmanager
import json
import itertools
import numpy as np
from math import radians, inf
from mathutils import Matrix, Vector
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple
import base64

@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w')# Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different

class training_samples():

    def __init__(self):
        self.root_mesh_dir = 'objects'
        self.camera_distance = - 6.0

        base64_matrix = []
        index = []
        names = []
        scale = []

        if "--objaverse" in sys.argv:
            self.root_path = sys.argv[sys.argv.index("--objaverse") + 1]
        if "--output" in sys.argv:
            self.output_path = sys.argv[sys.argv.index("--output") + 1]
        if "--camera_dist" in sys.argv:
            self.camera_distance = - float(sys.argv[sys.argv.index("--camera_dist") + 1])

        for arg in sys.argv[1:]:
            if arg == "--transform":
                base64_matrix.append(sys.argv[sys.argv.index("--transform") + 1])
                del sys.argv[sys.argv.index("--transform"):sys.argv.index("--transform") + 2]
            elif arg == "--object":
                names.append(sys.argv[sys.argv.index("--object") + 1])
                index.append(int(sys.argv[sys.argv.index("--object") + 2]))
                del sys.argv[sys.argv.index("--object"):sys.argv.index("--object") + 3]
            elif arg == "--scale":
                scale.append(float(sys.argv[sys.argv.index("--scale") + 1]))
                del sys.argv[sys.argv.index("--scale"):sys.argv.index("--scale") + 2]

        objects = os.listdir(self.root_mesh_dir)
        names = [name + '.json' for name in names]
        assert all([obj in objects for obj in names])
        if len(base64_matrix) > 0:
            self.objects = zip(names, index, base64_matrix, scale)
        else:
            self.objects = zip(names, index)


        if "--training" in sys.argv:
            print('training mode')
            self.extrat_training_renders()
        if "--SI-PnP" in sys.argv:
            print('SI-PnP mode')
            self.rotation = np.array([[1.0, 0.0, 0.0, 0.0],
                                      [0.0, -1.0, 0.0, 0.0],
                                      [0.0, 0.0, -1.0, 0.0],
                                      [0.0, 0.0, 0.0, 1.0]])
            self.extrat_SI_PnP_renders()
        if "--evaluate" in sys.argv:
            self.camera_distance *= 2 / 3
            self.extrat_transform_renders()

    @staticmethod
    def get_calibration_matrix_K_from_blender(mode='complete'):
        # https://mcarletti.github.io/articles/blenderintrinsicparams/
        scene = bpy.context.scene

        scale = scene.render.resolution_percentage / 100
        width = scene.render.resolution_x * scale  # px
        height = scene.render.resolution_y * scale  # px

        camdata = scene.camera.data

        if mode == 'simple':
            aspect_ratio = width / height
            K = np.zeros((3, 3), dtype=np.float32)
            K[0][0] = width / 2 / np.tan(camdata.angle / 2)
            K[1][1] = height / 2. / np.tan(camdata.angle / 2) * aspect_ratio
            K[0][2] = width / 2.
            K[1][2] = height / 2.
            K[2][2] = 1.
            K.transpose()

        return K

    @staticmethod
    def reset_scene() -> None:
        """Resets the scene to a clean state.

        Returns:
            None
        """
        # delete everything that isn't part of a camera or a light
        for obj in bpy.data.objects:
            if obj.type not in {"CAMERA", "LIGHT"}:
                bpy.data.objects.remove(obj, do_unlink=True)

        # delete all the materials
        for material in bpy.data.materials:
            bpy.data.materials.remove(material, do_unlink=True)

        # delete all the textures
        for texture in bpy.data.textures:
            bpy.data.textures.remove(texture, do_unlink=True)

        # delete all the images
        for image in bpy.data.images:
            bpy.data.images.remove(image, do_unlink=True)

    @staticmethod
    def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
        """Returns all meshes in the scene.

        Yields:
            Generator[bpy.types.Object, None, None]: Generator of all meshes in the scene.
        """
        for obj in bpy.context.scene.objects.values():
            if isinstance(obj.data, (bpy.types.Mesh)):
                yield obj

    @staticmethod
    def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
        """Returns all root objects in the scene.

        Yields:
            Generator[bpy.types.Object, None, None]: Generator of all root objects in the
                scene.
        """
        for obj in bpy.context.scene.objects.values():
            if not obj.parent:
                yield obj

    def scene_bbox(self,
            single_obj: Optional[bpy.types.Object] = None, ignore_matrix: bool = False,
                   exlude_obj: list = []) -> Tuple[Vector, Vector]:
        """Returns the bounding box of the scene.

        Taken from Shap-E rendering script
        (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

        Args:
            single_obj (Optional[bpy.types.Object], optional): If not None, only computes
                the bounding box for the given object. Defaults to None.
            ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
                to False.

        Raises:
            RuntimeError: If there are no objects in the scene.

        Returns:
            Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
        """
        bbox_min = (inf,) * 3
        bbox_max = (-inf,) * 3
        found = False
        for obj in self.get_scene_meshes() if single_obj is None else [single_obj]:
            if obj.name not in exlude_obj:
                found = True
                for coord in obj.bound_box:
                    coord = Vector(coord)
                    if not ignore_matrix:
                        coord = obj.matrix_world @ coord
                    bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                    bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

        if not found:
            raise RuntimeError("no objects in scene to compute bounding box for")

        return Vector(bbox_min), Vector(bbox_max)

    def find_floor(self,
            single_obj: Optional[bpy.types.Object] = None, ignore_matrix: bool = False):

        bbox = []
        for obj in self.get_scene_meshes() if single_obj is None else [single_obj]:
            bbox_array = []
            for coord in obj.bound_box:
                coord = Vector(coord)
                if not ignore_matrix:
                    coord = obj.matrix_world @ coord
                bbox_array.append(coord)
            bbox_array = np.stack(bbox_array)
            bbox.append(bbox_array)
        bbox = np.concatenate(bbox)

        min_coords = np.min(bbox, axis=0)
        max_coords = np.max(bbox, axis=0)
        bbox = np.array([min_coords,
                [max_coords[0], min_coords[1], min_coords[2]],
                [max_coords[0], max_coords[1], min_coords[2]],
                [min_coords[0], max_coords[1], min_coords[2]],
                [min_coords[0], min_coords[1], max_coords[2]],
                [max_coords[0], min_coords[1], max_coords[2]],
                max_coords,
                [min_coords[0], max_coords[1], max_coords[2]]])

        return bbox[[2, 3, 6, 7]]

    def extrat_transform_renders(self):
        self.reset_scene()
        camera_data = bpy.data.cameras.new("Camera")
        camera = bpy.data.objects.new("Camera", camera_data)

        C = bpy.context
        world = C.scene.world
        world.use_nodes = True
        enode = C.scene.world.node_tree.nodes.new("ShaderNodeTexEnvironment")
        enode.image = bpy.data.images.load("utils/rural_crossroads_8k.hdr")
        node_tree = C.scene.world.node_tree
        node_tree.links.new(enode.outputs['Color'], node_tree.nodes['Background'].inputs['Color'])
        bpy.context.scene.render.film_transparent = True

        scene = bpy.context.scene
        scene.use_nodes = True
        bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'GPU'
        scene.render.resolution_x = 512
        scene.render.resolution_y = 512

        names = []
        objects_name = []
        current_parts = []
        for i, (obj_name, idx, base64_matrix, scale) in enumerate(self.objects):
            if i > 0:
                current_parts += [o.name for o in self.get_scene_meshes()]

            binary_string = base64.b64decode(base64_matrix)
            transform_matrix = np.frombuffer(binary_string, dtype=np.float64).reshape((4, 4))

            obj_json_path = os.path.join(self.root_mesh_dir, obj_name)
            with open(obj_json_path, 'r') as f:
                concept = json.load(f)

            relative_path = concept['path'][idx]
            obj_path = os.path.join(self.root_path, relative_path.lstrip(os.sep))
            names.append(obj_name.replace('.json', ''))

            # import model
            with stdout_redirected():
                bpy.ops.import_scene.gltf(filepath=obj_path, merge_vertices=True)
            obj = bpy.context.selected_objects[0]
            obj.matrix_world = np.eye(4)

            obj.scale *= scale
            bpy.context.view_layer.update()

            obj.matrix_world = Matrix(transform_matrix) @ obj.matrix_world
            objects_name.append(obj.name)

        bpy.context.view_layer.update()
        bbox_min, bbox_max = self.scene_bbox()
        offset = -(bbox_min + bbox_max) / 2
        for name in objects_name:
            bpy.data.objects[name].matrix_world.translation += offset

        anglez = 10
        for anglex in range(-180, 181, 5):
            scene.camera = camera
            C.window.scene = scene
            C.scene.name = obj_path

            camera.location = (self.camera_distance * np.sin(anglex * np.pi / 180),
                               self.camera_distance * np.tan(anglez * np.pi / 180),
                               self.camera_distance * np.cos(anglex * np.pi / 180))
            camera.rotation_euler = ([radians(a) for a in (180 - anglez, anglex, 0)])

            C.scene.camera = camera
            C.scene.render.filepath = f"{self.output_path}/%d_%d____.png" % (anglex, anglez)
            with stdout_redirected():
                bpy.ops.render.render(write_still=True)

    def extrat_training_renders(self):
        self.reset_scene()

        for obj_name, idx in self.objects:
            obj_json_path = os.path.join(self.root_mesh_dir, obj_name)
            with open(obj_json_path, 'r') as f:
                concept = json.load(f)

            if idx > -1:
                concept['path'] = [concept['path'][idx]]

            for relative_path in concept['path']:

                obj_path = os.path.join(self.root_path, relative_path.lstrip(os.sep))
                obj_name = obj_name.replace('.json', '')
                file_name = os.path.basename(obj_path).replace('.glb', '')
                obj_dir = f"inputs/personalization/{obj_name}/{file_name}"

                if os.path.exists(obj_dir):
                    print(f'-------------- {obj_name.replace(".json", "")} -------------- SKIPPED')
                    continue
                else:
                    print(f'-------------- {obj_name.replace(".json", "")} --------------')

                camera_data = bpy.data.cameras.new("Camera")
                camera = bpy.data.objects.new("Camera", camera_data)

                C = bpy.context
                world = C.scene.world
                world.use_nodes = True
                enode = C.scene.world.node_tree.nodes.new("ShaderNodeTexEnvironment")
                enode.image = bpy.data.images.load("utils/rural_crossroads_8k.hdr")
                node_tree = C.scene.world.node_tree
                node_tree.links.new(enode.outputs['Color'], node_tree.nodes['Background'].inputs['Color'])
                bpy.context.scene.render.film_transparent = True

                scene = bpy.context.scene
                scene.use_nodes = True
                bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
                scene.render.engine = 'CYCLES'
                scene.cycles.device = 'GPU'
                scene.render.resolution_x = 2048
                scene.render.resolution_y = 2048

                # import model
                with stdout_redirected():
                    bpy.ops.import_scene.gltf(filepath=obj_path, merge_vertices=True)
                obj = bpy.context.selected_objects[0]
                obj.matrix_world = np.eye(4)

                bbox_min, bbox_max = self.scene_bbox()
                bbox_length = bbox_max - bbox_min
                scale = concept['x_size_cm'] / bbox_length[0] / 100
                obj.scale = obj.scale * scale

                # Apply scale to matrix_world.
                bpy.context.view_layer.update()
                bbox_min, bbox_max = self.scene_bbox()
                offset = -(bbox_min + bbox_max) / 2
                obj.matrix_world.translation += offset

                for anglex in range(-20, 21, 4):
                    for anglez in range(10, 31, 10):
                        scene.camera = camera
                        C.window.scene = scene
                        C.scene.name = obj_path

                        camera.location = (self.camera_distance * np.sin(anglex * np.pi / 180),
                                           self.camera_distance * np.tan(anglez * np.pi / 180),
                                           self.camera_distance * np.cos(anglex * np.pi / 180))
                        camera.rotation_euler = ([radians(a) for a in (180 - anglez, anglex, 0)])

                        C.scene.camera = camera
                        C.scene.render.filepath = f"{obj_dir}/img_%d_%d_.png" % (anglex, anglez)
                        with stdout_redirected():
                            bpy.ops.render.render(write_still=True)

                self.reset_scene()

    def extrat_SI_PnP_renders(self):
        self.reset_scene()

        for obj_name, idx in self.objects:
            obj_json_path = os.path.join(self.root_mesh_dir, obj_name)
            with open(obj_json_path, 'r') as f:
                concept = json.load(f)

            if idx > -1:
                concept['path'] = [concept['path'][idx]]

            for relative_path in concept['path']:

                obj_path = os.path.join(self.root_path, relative_path.lstrip(os.sep))
                obj_name = obj_name.replace('.json', '')
                file_name = os.path.basename(obj_path).replace('.glb', '')
                obj_dir = f"inputs/SI-PnP/{obj_name}/{file_name}"

                if os.path.exists(obj_dir):
                    print(f'-------------- {obj_name.replace(".json", "")} -------------- SKIPPED')
                    continue
                else:
                    print(f'-------------- {obj_name.replace(".json", "")} --------------')

                camera_data = bpy.data.cameras.new("Camera")
                camera = bpy.data.objects.new("Camera", camera_data)

                C = bpy.context
                world = C.scene.world
                world.use_nodes = True
                enode = C.scene.world.node_tree.nodes.new("ShaderNodeTexEnvironment")
                enode.image = bpy.data.images.load("utils/rural_crossroads_8k.hdr")
                node_tree = C.scene.world.node_tree
                node_tree.links.new(enode.outputs['Color'], node_tree.nodes['Background'].inputs['Color'])
                bpy.context.scene.render.film_transparent = True

                to_save_camera = {}
                scene = bpy.context.scene
                scene.use_nodes = True
                bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
                scene.render.engine = 'CYCLES'
                scene.cycles.device = 'GPU'
                scene.render.resolution_x = 512
                scene.render.resolution_y = 512

                links = bpy.context.scene.node_tree.links
                nodes = bpy.context.scene.node_tree.nodes
                for n in nodes:
                    nodes.remove(n)
                render_layers = nodes.new('CompositorNodeRLayers')
                depth_file_output = nodes.new(type="CompositorNodeOutputFile")
                depth_file_output.format.file_format = 'OPEN_EXR'
                depth_file_output.label = 'Depth Output'
                depth_file_output.base_path = ''
                depth_file_output.file_slots[0].use_node_format = True

                links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])

                # import model
                with stdout_redirected():
                    bpy.ops.import_scene.gltf(filepath=obj_path, merge_vertices=True)
                obj = bpy.context.selected_objects[0]
                obj.matrix_world = np.eye(4)

                # Apply scale to matrix_world.
                bbox_min, bbox_max = self.scene_bbox()
                bbox_length = bbox_max - bbox_min
                scale = concept['x_size_cm'] / bbox_length[0] / 100
                obj.scale *= scale
                to_save_camera['scale'] = scale
                bpy.context.view_layer.update()

                # find the floor points
                to_save_camera['floor'] = self.find_floor()

                # Apply centering
                bbox_min, bbox_max = self.scene_bbox()
                offset = -(bbox_min + bbox_max) / 2
                obj.matrix_world.translation += offset
                bpy.context.view_layer.update()

                obj.rotation_mode = "XYZ"

                scene.camera = camera
                C.window.scene = scene
                C.scene.name = obj_path
                anglez_ = 0; anglex_ = 0
                camera.location = (0.6 * self.camera_distance * np.sin(anglex_ * np.pi / 180),
                                   0.6 * self.camera_distance * np.tan(anglez_ * np.pi / 180),
                                   0.6 * self.camera_distance * np.cos(anglex_ * np.pi / 180))
                camera.rotation_euler = ([radians(a) for a in (180 - anglez_, anglex_, 0)])
                C.scene.camera = camera

                anglez = 30
                for anglex in range(-50, 60, 10):

                    object_rotation = (radians(anglez), radians(anglex), 0)
                    obj.rotation_euler = object_rotation
                    bpy.context.view_layer.update()

                    C.scene.render.filepath = f"{obj_dir}/img_%d_%d_.png" % (anglex, anglez)
                    depth_file_output.file_slots[0].path = f"{obj_dir}/depth_%d_%d_depth.png" % (anglex, anglez)
                    with stdout_redirected():
                        bpy.ops.render.render(write_still=True)

                    intrinsics = np.array(self.get_calibration_matrix_K_from_blender(mode='simple'))
                    to_save_camera["K_%03d_%03d" % (anglex, anglez)] = intrinsics

                    modelview_matrix = camera.matrix_world.inverted()
                    modelview_matrix = np.array(modelview_matrix)
                    M = self.rotation @ modelview_matrix @ np.array(obj.matrix_world)
                    to_save_camera["M_%03d_%03d" % (anglex, anglez)] = M

                np.savez(f"{obj_dir}/cameras.npz", **to_save_camera)
                self.reset_scene()


if __name__ == "__main__":
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1

    training_samples()

