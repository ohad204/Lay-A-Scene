import torch.nn as nn
from utils.personalization import SpatialDreambooth


class StableDiffusion(nn.Module):
    def __init__(self, device, opt, concepts_list, combinations_dir):
        super().__init__()
        self.device = device
        self.save_path = opt.save_path
        self.combinations_dir = combinations_dir
        print(f'[INFO] loading stable diffusion...')

        # Create model
        personalization = SpatialDreambooth(concepts_list, experiment_name=opt.expr_name,
                                       class_prompt=opt.scene_description,
                                       training_data_dir=self.combinations_dir,
                                       load_ckpt=opt.load_personalization,
                                       save_path=self.save_path,
                                       device=self.device)
        self.pipe = personalization.get_pipeline()
        del personalization

        self.pipe.set_progress_bar_config(disable=True)
        print(f'[INFO] loaded stable diffusion!')
