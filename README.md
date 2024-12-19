# Lay-A-Scene
This is the official implementation of the paper **Lay-A-Scene: Personalized 3D Object Arrangement Using Text-to-Image Priors**. we propose a method find a plausible arrangement of these objects in a scene, based on stable-diffusion model.
<br/>
> **Lay-A-Scene: Personalized 3D Object Arrangement Using Text-to-Image Priors**<br>
> Ohad Rahamim<sup>1</sup>, Hilit Segev<sup>1</sup>, Idan Achituve<sup>1</sup>, Yuval Atzmon<sup>2</sup>, Yoni Kasten<sup>2</sup>, Gal Chechik<sup>1,2</sup> <br>
> <sup>1</sup>Bar Ilan University, <sup>2</sup>NVIDIA research

>**Abstract**: <br>
>     Generating 3D visual scenes is at the forefront for visual generative AI, but current 3D generation techniques struggle with generating scenes with multiple high resolution objects. Here we introduce **Lay-A-Scene** a new setup for 3D scene generation: **3D Arrangement**. Given a set of 3D objects, the task is to find a plausible arrangement  of these objects in a scene. We address this task by leveraging pretrained text-to image models: we first personalize the model to teach it to generate a scene that contains given objects. Then, we infer the 3D poses and arrangement of objects from the generated image, by finding a consistent projection of objects to the 2D scene. This method avoids slow optimization of object shapes using SDS loss. We evaluate the quality of Lay-A-Scene using 3D objects from Objverse and human raters, and find that it often generates coherent and feasible 3D object arrangements.

## Setup

### Environment

Download the code and go the folder.
```bash
git clone https://github.com/ohad204/Lay-A-Scene.git
cd Lay-A-Scene
```

Install the conda virtual environment:
```bash
conda env create -f environment.yml
conda activate Lay-A-Scene
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
```

Install Blender

```bash
wget https://download.blender.org/release/Blender3.6/blender-3.6.5-linux-x64.tar.xz
tar -xf blender-3.6.5-linux-x64.tar.xz
rm blender-3.6.5-linux-x64.tar.xz
```

## Evaluation

To evaluate the models run:
```bash
export OPENCV_IO_ENABLE_OPENEXR=1; python main.py --objects sofa chair --objects_idx 0 0 --transform_optimization --extract_renders
```

`--objects`: the objects from `objects` dictionary.

`--objects_idx`: if several objects assigned to each class (i.e. `sofa.json`), we can select which one to address.

`--transform_optimization`: finding common plane and Adjust the camera accordingly.

`--extract_renders`: let blender compute renders and save (skipped if already exists ).

## Results
Here are some sample results. Please read our paper for more!

<div style="display: flex;">
    <img src="gifs/sofa_chair_1.gif" alt="Sofa Chair GIF" width="200" height="200">
    <img src="gifs/sofa_chair_2.gif" alt="Sofa Chair GIF" width="200" height="200">
    <img src="gifs/sofa_table_1.gif" alt="Sofa Chair GIF" width="200" height="200">
</div>