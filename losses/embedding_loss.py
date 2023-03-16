import torch
from losses import lpips
from losses.style.style_loss import StyleLoss
import PIL
import os


class EmbeddingLossBuilder(torch.nn.Module):
    def __init__(self, opt):
        super(EmbeddingLossBuilder, self).__init__()

        self.opt = opt
        self.parsed_loss = [[opt.l2_lambda, 'l2'], [opt.percept_lambda, 'percep']]
        self.l2 = torch.nn.MSELoss()
        if opt.device == 'cuda':
            use_gpu = True
        else:
            use_gpu = False
        self.percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=use_gpu)
        self.percept.eval()
        # self.percept = VGGLoss()




    def _loss_l2(self, gen_im, ref_im, **kwargs):
        return self.l2(gen_im, ref_im)


    def _loss_lpips(self, gen_im, ref_im, **kwargs):

        
        return self.percept(gen_im, ref_im).sum()



    def forward(self, ref_im_H,ref_im_L, gen_im_H, gen_im_L):

        loss = 0
        loss_fun_dict = {
            'l2': self._loss_l2,
            'percep': self._loss_lpips,
        }
        losses = {}
        for weight, loss_type in self.parsed_loss:
            if loss_type == 'l2':
                var_dict = {
                    'gen_im': gen_im_H,
                    'ref_im': ref_im_H,
                }
            elif loss_type == 'percep':
                var_dict = {
                    'gen_im': gen_im_L,
                    'ref_im': ref_im_L,
                }
            tmp_loss = loss_fun_dict[loss_type](**var_dict)
            losses[loss_type] = tmp_loss
            loss += weight*tmp_loss
        return loss, losses


class BaldHairEmbeddingLossBuilder(torch.nn.Module):
    def __init__(self, opt):
        super(BaldHairEmbeddingLossBuilder, self).__init__()

        self.opt = opt
        self.parsed_loss = [[opt.l2_lambda, 'l2'], [opt.percept_lambda, 'percep']]
        self.l2 = torch.nn.MSELoss()
        if opt.device == 'cuda':
            use_gpu = True
        else:
            use_gpu = False
        self.percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=use_gpu)
        self.percept.eval()
        # self.percept = VGGLoss()

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.style = StyleLoss(distance="l2", VGG16_ACTIVATIONS_LIST=[3, 8, 15, 22], normalize=False).to(opt.device)
        self.style.eval()

        tmp = torch.zeros(16).to(opt.device)
        tmp[0] = 1
        self.cross_entropy_wo_background = torch.nn.CrossEntropyLoss(weight=1 - tmp)
        self.cross_entropy_only_background = torch.nn.CrossEntropyLoss(weight=tmp)

    def _loss_l2(self, gen_im, ref_im, **kwargs):
        return self.l2(gen_im, ref_im)


    def _loss_lpips(self, gen_im, ref_im, **kwargs):
        return self.percept(gen_im, ref_im).sum()

    def cross_entropy_loss(self, down_seg, target_mask):
        loss = self.opt.ce_lambda * self.cross_entropy(down_seg, target_mask)
        return loss


    def style_loss(self, im1, im2, mask1, mask2):
        loss = self.opt.style_lambda * self.style(im1 * mask1, im2 * mask2, mask1=mask1, mask2=mask2)
        return loss

    def cross_entropy_loss_wo_background(self, down_seg, target_mask):
        loss = self.opt.ce_lambda * self.cross_entropy_wo_background(down_seg, target_mask)
        return loss

    def cross_entropy_loss_only_background(self, down_seg, target_mask):
        loss = self.opt.ce_lambda * self.cross_entropy_only_background(down_seg, target_mask)
        return loss

    def forward(self, ref_im_H,ref_im_L, gen_im_H, gen_im_L, down_seg = None, wo_bg_seg = None):

        loss = 0
        loss_fun_dict = {
            'l2': self._loss_l2,
            'percep': self._loss_lpips,
        }
        losses = {}
        for weight, loss_type in self.parsed_loss:
            if loss_type == 'l2':
                var_dict = {
                    'gen_im': gen_im_H,
                    'ref_im': ref_im_H,
                }
            elif loss_type == 'percep':
                var_dict = {
                    'gen_im': gen_im_L,
                    'ref_im': ref_im_L,
                }
            tmp_loss = loss_fun_dict[loss_type](**var_dict)
            losses[loss_type] = tmp_loss
            loss += weight*tmp_loss
        
        if down_seg is not None and wo_bg_seg is not None:
#             print('qqqq')
            ce_loss = self.cross_entropy_loss_wo_background(down_seg, wo_bg_seg)
            losses["ce_loss"] = ce_loss
            loss += ce_loss
#         ce_loss += self.cross_entropy_loss_only_background(down_seg, seg)
        
        return loss, losses
