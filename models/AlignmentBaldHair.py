import torch
from torch import nn
from models.Net import Net
import numpy as np
import os
from functools import partial
from utils.bicubic import BicubicDownSample
from tqdm import tqdm
import PIL
import torchvision
from PIL import Image
from utils.data_utils import convert_npy_code
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from losses.align_loss import AlignLossBuilder
import torch.nn.functional as F
import cv2
from scipy.ndimage import label
from utils.data_utils import load_FS_latent
from utils.seg_utils import save_vis_mask, vis_seg, get_color_index
from utils.model_utils import download_weight
from utils.data_utils import cuda_unsqueeze
from utils.image_utils import dilate_erosion_mask_tensor

toPIL = torchvision.transforms.ToPILImage()

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
thickness = 2
color = (0, 0, 255)
text_x = 20
text_y = 100


class Alignment(nn.Module):

    def __init__(self, opts, net=None):
        super(Alignment, self).__init__()
        self.opts = opts
        if not net:
            self.net = Net(self.opts)
        else:
            self.net = net

        self.load_segmentation_network()
        self.load_downsampling()
        self.setup_align_loss_builder()

    def load_segmentation_network(self):
        self.seg = BiSeNet(n_classes=16)
        self.seg.to(self.opts.device)

        if not os.path.exists(self.opts.seg_ckpt):
            download_weight(self.opts.seg_ckpt)
        self.seg.load_state_dict(torch.load(self.opts.seg_ckpt))
        for param in self.seg.parameters():
            param.requires_grad = False
        self.seg.eval()

    def load_downsampling(self):
        self.downsample = BicubicDownSample(factor=self.opts.size // 512)
        self.downsample_256 = BicubicDownSample(factor=self.opts.size // 256)

    def setup_align_loss_builder(self):
        self.loss_builder = AlignLossBuilder(self.opts)

    def create_bald_segmentation_mask(self, img_path, sign, save_intermediate=True):
        # 1. load image from path
        img = self.preprocess_img(img_path)

        # 2. segmentation
        down_seg, _, _ = self.seg(img) # shape: torch.Size([1, 16, 512, 512])
        seg = torch.argmax(down_seg, dim=1).long() # shape: torch.Size([1, 512, 512])

        # 3. remove hair region
        seg1 = torch.where(seg  == 10, 0, seg) # shape: torch.Size([512, 512]), 0~15

        # 4. background segment is already in original seg img

        # 5. optimization
        optimizer_align, latent_align = self.setup_align_optimizer()
        latent_end = latent_align[:, 6:, :].clone().detach()
        pbar = tqdm(range(80), desc='Create Target Mask Step1', leave=False)
        for step in pbar:
            optimizer_align.zero_grad()
            latent_in = torch.cat([latent_align[:, :6, :], latent_end], dim=1)
            down_seg, _ = self.create_down_seg(latent_in)

            ce_loss = self.loss_builder.cross_entropy_loss_wo_background(down_seg, seg1)
            ce_loss += self.loss_builder.cross_entropy_loss_only_background(down_seg, seg)

            loss = ce_loss
            loss.backward()
            optimizer_align.step()

        # 6. largest region to mask
        gen_seg = torch.argmax(down_seg, dim=1).long() # shape: torch.Size([1, 512, 512])
        gen_seg_np = gen_seg.squeeze(0).detach().cpu().numpy()

        # Count the number of regions of 255
        labeled_array, num_labels = label(gen_seg_np == 10)
        # Find the largest region
        largest_region_label = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1
        # Make mask
        inpainting_mask = np.where(labeled_array == largest_region_label, 255, 0).astype(np.uint8)

        # 7. other region to background index(0)
        gen_seg_np = np.where(gen_seg_np == 10, 0 , gen_seg_np)
        color_seg = vis_seg(gen_seg_np)

        # 8. inpainting with opencv
        bald_seg = cv2.inpaint(color_seg, inpainting_mask, 3, cv2.INPAINT_NS)
        bald_seg_tensor = torch.from_numpy(bald_seg)

        # 9. Convert the segmented image to an index image
        index_image = torch.zeros_like(gen_seg.squeeze(0))
        color = torch.from_numpy(get_color_index())
        for i in range(len(color)):
            mask = torch.all(bald_seg_tensor == color[i], axis=-1)
            index_image[mask] = i
            
        if save_intermediate:
            save_vis_mask(img_path, img_path, sign, self.opts.output_dir, index_image.cpu())
            
        return index_image.unsqueeze(0), torch.where(index_image==10, 1, 0).unsqueeze(0).unsqueeze(0).float(), torch.where(seg  == 10, 1, 0), torch.zeros_like(seg1)
    

    def get_img_name(self, path):
        name, _ = os.path.basename(path).split('.')
        return name
        
    def preprocess_img(self, img_path):
        im = torchvision.transforms.ToTensor()(Image.open(img_path))[:3].unsqueeze(0).to(self.opts.device)
        im = (self.downsample(im).clamp(0, 1) - seg_mean) / seg_std
        return im

    def setup_align_optimizer(self, latent_path=None):
        if latent_path:
            latent_W = torch.from_numpy(convert_npy_code(np.load(latent_path))).to(self.opts.device).requires_grad_(True)
        else:
            latent_W = self.net.latent_avg.reshape(1, 1, 512).repeat(1, 18, 1).clone().detach().to(self.opts.device).requires_grad_(True)

        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }

        optimizer_align = opt_dict[self.opts.opt_name]([latent_W], lr=self.opts.learning_rate)

        return optimizer_align, latent_W



    def create_down_seg(self, latent_in):
        gen_im, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False,
                                       start_layer=0, end_layer=8)
        gen_im_0_1 = (gen_im + 1) / 2

        # get hair mask of synthesized image
        im = (self.downsample(gen_im_0_1) - seg_mean) / seg_std
        down_seg, _, _ = self.seg(im)
        return down_seg, gen_im


    def dilate_erosion(self, free_mask, device, dilate_erosion=5):
        free_mask = F.interpolate(free_mask.cpu(), size=(256, 256), mode='nearest').squeeze()
        free_mask_D, free_mask_E = cuda_unsqueeze(dilate_erosion_mask_tensor(free_mask, dilate_erosion=dilate_erosion), device)
        return free_mask_D, free_mask_E

    def align_images(self, img_path1, sign='realistic', align_more_region=False, smooth=5,
                     save_intermediate=True):

        device = self.opts.device
        output_dir = self.opts.output_dir
        target_mask, hair_mask_target, hair_mask1, hair_mask2 = self.create_bald_segmentation_mask(
            img_path1, 
            sign = sign, 
            save_intermediate=save_intermediate
        )
    
        im_name_1 = os.path.splitext(os.path.basename(img_path1))[0]
#         im_name_2 = os.path.splitext(os.path.basename(img_path2))[0]

        latent_FS_path_1 = os.path.join(output_dir, 'FS', f'{im_name_1}.npz')
#         latent_FS_path_2 = os.path.join(output_dir, 'FS', f'{im_name_2}.npz')

        latent_1, latent_F_1 = load_FS_latent(latent_FS_path_1, device)
#         latent_2, latent_F_2 = load_FS_latent(latent_FS_path_2, device)

        latent_W_path_1 = os.path.join(output_dir, 'W+', f'{im_name_1}.npy')
#         latent_W_path_2 = os.path.join(output_dir, 'W+', f'{im_name_2}.npy')



        pbar = tqdm(range(self.opts.align_steps1), desc='Align Step 1', leave=False)
        os.makedirs('output/loss/Align1/', exist_ok=True)
        log_align1_loss = ""
        
        # Create a VideoWriter object to write the video to a file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_rate = 60
        video_writer = cv2.VideoWriter(f'output/loss/Align1/{im_name_1}.mp4', fourcc, frame_rate, (1024, 1024))
        last_step = 0
        
        optimizer_align, latent_align_1 = self.setup_align_optimizer(latent_W_path_1)
        with torch.no_grad():
            tmp_latent_in = torch.cat([latent_align_1[:, :6, :], latent_1[:, 6:, :]], dim=1)
            down_seg_tmp, I_Structure_Style_changed = self.create_down_seg(tmp_latent_in)

            current_mask_tmp = torch.argmax(down_seg_tmp, dim=1).long()
            HM_Structure = torch.where(current_mask_tmp == 10, torch.ones_like(current_mask_tmp),
                                       torch.zeros_like(current_mask_tmp))
            HM_Structure = F.interpolate(HM_Structure.float().unsqueeze(0), size=(256, 256), mode='nearest')
        
        for step in pbar:
            optimizer_align.zero_grad()
            latent_in = torch.cat([latent_align_1[:, :6, :], latent_1[:, 6:, :]], dim=1)
            down_seg, gen_im = self.create_down_seg(latent_in)
            
            tensor_image = ((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1)
            # convert the tensor to a PIL image
            pil_image = toPIL(tensor_image)
            # convert the PIL image to an OpenCV image
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            # Add text to the image
            text = f"{step}"
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.putText(opencv_image, text, (text_x, text_y), font, font_scale, color, thickness)                
            video_writer.write(opencv_image)

            loss_dict = {}
            ##### Cross Entropy Loss
            ce_loss = self.loss_builder.cross_entropy_loss(down_seg, target_mask)
            loss_dict["ce_loss"] = ce_loss.item()
            loss = ce_loss
            
            #### Style Loss
            Current_Mask = torch.argmax(down_seg, dim=1).long()
            HM_G_512 = torch.where(Current_Mask == 10, torch.ones_like(Current_Mask),
                                   torch.zeros_like(Current_Mask)).float().unsqueeze(0)
            HM_G = F.interpolate(HM_G_512, size=(256, 256), mode='nearest')
            H1_region = self.downsample_256(I_Structure_Style_changed) * HM_Structure
            H2_region = self.downsample_256(gen_im) * HM_G
            
            style_loss = self.loss_builder.style_loss(H1_region, H2_region, mask1=HM_Structure, mask2=HM_G)
            loss_dict["style_loss"] = style_loss.item()
            loss += style_loss
            


            for key in loss_dict.keys():
                try:
                    log_align1_loss += f"{loss_dict[key].item()},"
                except:
                    log_align1_loss += f"{loss_dict[key]},"
            log_align1_loss += '\n'
            
            loss.backward()
            optimizer_align.step()
        video_writer.release()
        
        with open(f'output/loss/Align1/{im_name_1}.txt', 'w') as log2: 
            for key in loss_dict.keys():
                log2.write(f"{key},")
            log2.write(f"\n{log_align1_loss}\n")
        
        intermediate_align, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False,
                                                   start_layer=0, end_layer=3)
        intermediate_align = intermediate_align.clone().detach()

        ##############################################
        
        latent_F_out_new, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False,
                                                 start_layer=0, end_layer=3)
        latent_F_out_new = latent_F_out_new.clone().detach()

        free_mask = 1 - (1 - hair_mask1.unsqueeze(0)) * (1 - hair_mask_target)

        ##############################
        free_mask, _ = self.dilate_erosion(free_mask, device, dilate_erosion=smooth)
        ##############################

        free_mask_down_32 = F.interpolate(free_mask.float(), size=(32, 32), mode='bicubic')[0]
        interpolation_low = 1 - free_mask_down_32


        latent_F_mixed = intermediate_align + interpolation_low.unsqueeze(0) * (
                latent_F_1 - intermediate_align)

        if not align_more_region:
            free_mask = hair_mask_target
            ##########################
            _, free_mask = self.dilate_erosion(free_mask, device, dilate_erosion=smooth)
            ##########################
            free_mask_down_32 = F.interpolate(free_mask.float(), size=(32, 32), mode='bicubic')[0]
            interpolation_low = 1 - free_mask_down_32


        latent_F_mixed = latent_F_out_new + interpolation_low.unsqueeze(0) * (
                latent_F_mixed - latent_F_out_new)

        free_mask = F.interpolate((hair_mask2.unsqueeze(0) * hair_mask_target).float(), size=(256, 256), mode='nearest').cuda()
        ##########################
        _, free_mask = self.dilate_erosion(free_mask, device, dilate_erosion=smooth)
        ##########################
        free_mask_down_32 = F.interpolate(free_mask.float(), size=(32, 32), mode='bicubic')[0]
        interpolation_low = 1 - free_mask_down_32

        latent_F_mixed = latent_F_1 + interpolation_low.unsqueeze(0) * (
                latent_F_mixed - latent_F_1)

        gen_im, _ = self.net.generator([latent_1], input_is_latent=True, return_latents=False, start_layer=4,
                                       end_layer=8, layer_in=latent_F_mixed)
        self.save_align_results(im_name_1, sign, gen_im, latent_1, latent_F_mixed,
                                save_intermediate=save_intermediate)

    def save_align_results(self, im_name_1, sign, gen_im, latent_in, latent_F, save_intermediate=True):

        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))

        latent_path = os.path.join(self.opts.output_dir, '{}.npz'.format(im_name_1))
        if save_intermediate:
            image_path = os.path.join(self.opts.output_dir, '{}.png'.format(im_name_1))
            save_im.save(image_path)
        np.savez(latent_path, latent_in=latent_in.detach().cpu().numpy(), latent_F=latent_F.detach().cpu().numpy())


