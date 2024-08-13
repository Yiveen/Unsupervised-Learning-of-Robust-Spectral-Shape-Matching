import torch
import torch.nn.functional as F

from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import nn_query, fmap2pointmap


@MODEL_REGISTRY.register()
class FMNetModel(BaseModel):
    def __init__(self, opt):
        self.with_refine = opt.get('refine', -1)
        self.partial = opt.get('partial', False)
        self.non_isometric = opt.get('non-isometric', False)
        if self.with_refine > 0:
            opt['is_train'] = True
        super(FMNetModel, self).__init__(opt)

    def feed_data(self, data):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # feature extractor for mesh
        # feat_x = self.networks['feature_extractor'](data_x['verts'], data_x['faces'])  # [B, Nx, C]
        # feat_y = self.networks['feature_extractor'](data_y['verts'], data_y['faces'])  # [B, Ny, C]
        feat_x = self.networks['feature_extractor'](data_x['verts'], None)  # [B, Nx, C]
        feat_y = self.networks['feature_extractor'](data_y['verts'], None)  # [B, Ny, C]

        # get spectral operators
        evals_x = data_x['evals']
        evals_y = data_y['evals']
        evecs_x = data_x['evecs']
        evecs_y = data_y['evecs']
        evecs_trans_x = data_x['evecs_trans']  # [B, K, Nx]
        evecs_trans_y = data_y['evecs_trans']  # [B, K, Ny]

        # # Assuming temp is your tensor sliced from data_x['dino_feat'][:, :, :10]
        # dino_basis_x = data_x['dino_feat'][:,:,:10]
        # dino_basis_x = torch.flip(dino_basis_x, dims=[2])  # Reversing along the third dimension
        # dino_basis_y = data_y['dino_feat'][:,:,:10]
        # dino_basis_y = torch.flip(dino_basis_y, dims=[2])  # Reversing along the third dimension

        # dino_evals_x = data_x['dino_eval']
        # dino_evals_x = torch.flip(dino_evals_x, dims=[1])
        # # print('dino_evals', dino_evals_x)
        # # print('evecs_x',evals_x)
        # # print('evecs_x_shape',evals_x.shape)
        # dino_evals_y = data_y['dino_eval']
        # dino_evals_y = torch.flip(dino_evals_y, dims=[1])
        # # print('dino_evals_y', dino_evals_y.shape)
        
        
        gl_basis_x = data_x['gl_feat']
        gl_basis_x = torch.flip(gl_basis_x, dims=[2])  # Reversing along the third dimension
        gl_basis_y = data_y['gl_feat']
        gl_basis_y = torch.flip(gl_basis_y, dims=[2])  # Reversing along the third dimension

        gl_evals_x = data_x['gl_eval']
        gl_evals_x = torch.flip(gl_evals_x, dims=[1])
        # print('dino_evals', dino_evals_x)
        # print('evecs_x',evals_x)
        # print('evecs_x_shape',evals_x.shape)
        gl_evals_y = data_y['gl_eval']
        gl_evals_y = torch.flip(gl_evals_y, dims=[1])
        # print('dino_evals_y', dino_evals_y.shape)

        gl_trans_x = gl_basis_x.transpose(2, 1)
        gl_trans_y = gl_basis_y.transpose(2, 1)
        
        # combined_basis_x = torch.concatenate((gl_basis_x, dino_basis_x), dim=2)
        # combined_basis_y = torch.concatenate((gl_basis_y, dino_basis_y), dim=2)
        # # print(combined_basis_x.shape)
        
        # combined_trans_x = combined_basis_x.transpose(2, 1)
        # combined_trans_y = combined_basis_y.transpose(2, 1)
        # # print(combined_trans_x.shape)
        
        # combined_eval_x = torch.cat((gl_evals_x, gl_evals_x[:, -1].unsqueeze(1).expand(-1, 10)), dim=1) 
        # combined_eval_y = torch.cat((gl_evals_y, gl_evals_y[:, -1].unsqueeze(1).expand(-1, 10)), dim=1) 
        # print(combined_eval_x.shape)

        # Cxy, Cyx = self.networks['fmap_net'](feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)
        # Cxy, Cyx = self.networks['fmap_net'](feat_x, feat_y, dino_evals_x, dino_evals_y, dino_trans_x, dino_trans_y)
        Cxy, Cyx = self.networks['fmap_net'](feat_x, feat_y, gl_evals_x, gl_evals_y, gl_trans_x, gl_trans_y)
        # Cxy, Cyx = self.networks['fmap_net'](feat_x, feat_y, combined_eval_x, combined_eval_y, combined_trans_x, combined_trans_y)


        # self.loss_metrics = self.losses['surfmnet_loss'](Cxy, Cyx, evals_x, evals_y)
        # self.loss_metrics = self.losses['surfmnet_loss'](Cxy, Cyx, dino_evals_x, dino_evals_y)
        self.loss_metrics = self.losses['surfmnet_loss'](Cxy, Cyx, gl_evals_x, gl_evals_y)
        # self.loss_metrics = self.losses['surfmnet_loss'](Cxy, Cyx, gl_evals_x, gl_evals_y)


        Pxy, Pyx = self.compute_permutation_matrix(feat_x, feat_y, bidirectional=True)

        # compute C
        # Cxy_est = torch.bmm(evecs_trans_y, torch.bmm(Pyx, evecs_x))
        # Cxy_est = torch.bmm(dino_trans_y, torch.bmm(Pyx, dino_basis_x))
        Cxy_est = torch.bmm(gl_trans_y, torch.bmm(Pyx, gl_basis_x))
        # Cxy_est = torch.bmm(combined_trans_y, torch.bmm(Pyx, combined_basis_x))


        # print('Cxy',Cxy.shape)
        # print('Cxy_est',Cxy_est.shape)

        self.loss_metrics['l_align'] = self.losses['align_loss'](Cxy, Cxy_est)
        if not self.partial:
            # Cyx_est = torch.bmm(evecs_trans_x, torch.bmm(Pxy, evecs_y))
            # Cyx_est = torch.bmm(dino_trans_x, torch.bmm(Pyx, dino_basis_y))
            Cyx_est = torch.bmm(gl_trans_x, torch.bmm(Pyx, gl_basis_y))
            # Cyx_est = torch.bmm(combined_trans_x, torch.bmm(Pyx, combined_basis_y))

            self.loss_metrics['l_align'] += self.losses['align_loss'](Cyx, Cyx_est)

        if 'dirichlet_loss' in self.losses:
            Lx, Ly = data_x['L'], data_y['L']
            verts_x, verts_y = data_x['verts'], data_y['verts']
            self.loss_metrics['l_d'] = self.losses['dirichlet_loss'](torch.bmm(Pxy, verts_y), Lx) + \
                                       self.losses['dirichlet_loss'](torch.bmm(Pyx, verts_x), Ly)

    def validate_single(self, data, timer):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # get previous network state dict
        if self.with_refine > 0:
            state_dict = {'networks': self._get_networks_state_dict()}

        # start record
        timer.start()

        # test-time refinement
        if self.with_refine > 0:
            self.refine(data)

        # feature extractor
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x.get('faces'))
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y.get('faces'))

        # get spectral operators
        evecs_x = data_x['evecs'].squeeze()
        evecs_y = data_y['evecs'].squeeze()
        evecs_trans_x = data_x['evecs_trans'].squeeze()
        evecs_trans_y = data_y['evecs_trans'].squeeze()

        # dino_basis_x = data_x['dino_feat'].squeeze()
        # dino_basis_y = data_y['dino_feat'].squeeze()
        # dino_trans_x = dino_basis_x.transpose(1, 0)
        # dino_trans_y = dino_basis_y.transpose(1, 0)

        # if self.non_isometric:
        if True:
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)

            # nearest neighbour query
            p2p = nn_query(feat_x, feat_y).squeeze()

            # compute Pyx from functional map
            Cxy = evecs_trans_y @ evecs_x[p2p]
            Pyx = evecs_y @ Cxy @ evecs_trans_x
        else:
            # compute Pxy
            Pyx = self.compute_permutation_matrix(feat_y, feat_x, bidirectional=False).squeeze()

            Cxy = evecs_trans_y @ (Pyx @ evecs_x)
            # convert functional map to point-to-point map
            p2p = fmap2pointmap(Cxy, evecs_x, evecs_y)
            # compute Pyx from functional map
            Pyx = evecs_y @ Cxy @ evecs_trans_x

        #             Cxy = dino_trans_y @ (Pyx @ dino_basis_x)
        #             # convert functional map to point-to-point map
        #             p2p = fmap2pointmap(Cxy, dino_basis_x, dino_basis_y)

        #             # compute Pyx from functional map
        #             Pyx = dino_basis_y @ Cxy @ dino_trans_x
        # finish record
        timer.record()

        # resume previous network state dict
        if self.with_refine > 0:
            self.resume_model(state_dict, net_only=True, verbose=False)
        return p2p, Pyx, Cxy

    def compute_permutation_matrix(self, feat_x, feat_y, bidirectional=False, normalize=True):
        if normalize:
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)
        similarity = torch.bmm(feat_x, feat_y.transpose(1, 2))

        # sinkhorn normalization
        Pxy = self.networks['permutation'](similarity)

        if bidirectional:
            Pyx = self.networks['permutation'](similarity.transpose(1, 2))
            return Pxy, Pyx
        else:
            return Pxy

    def refine(self, data):
        self.networks['permutation'].hard = False
        self.networks['fmap_net'].bidirectional = True

        with torch.set_grad_enabled(True):
            for _ in range(self.with_refine):
                self.feed_data(data)
                self.optimize_parameters()

        self.networks['permutation'].hard = True
        self.networks['fmap_net'].bidirectional = False

    @torch.no_grad()
    def validation(self, dataloader, tb_logger, update=True):
        # change permutation prediction status
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = True
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = False
        super(FMNetModel, self).validation(dataloader, tb_logger, update)
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = False
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = True
