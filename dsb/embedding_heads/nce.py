from dsb.dependencies import *

from dsb.utils import torchify, untorchify, compute_output_shape
from dsb.agents.utils import update_target_network
import dsb.builder as builder
from dsb.builder import module_reset_parameters


class NCE(nn.Module):
    # this implementation follows SimCLR (but in batch, negatives do not include pairs, just other anchors)
    # and MoCo2 (without negatives queue)

    # Noise Contrastive Estimation
    # aka InfoNCE
    # aka Contrastive Predictive Coding
    # aka NT-XentLoss

    # see Table 2, https://arxiv.org/pdf/2002.05709.pdf#page=6

    def __init__(
        self,
        obs_space,
        encoder_network_params=[],
        embedding_dim=64,  # latent dim
        optim_params=dict(cls='Adam', lr=1e-3),
        optimize_interval=1,
        detach_embedding=False,  # if True, detach so other losses will not update encoder
        #
        pair_type='sampled_pair',
        similarity='bilinear',
        temperature=0.07,
        use_projection_head=True,
        projection_head_layers=2,
        normalize_projected=False,
        normalize_logits=True,
        detach_projected_pair=False,
        aug_params=None,
        aug_pair_only=False,
        detach_augmented=True,
        use_target_for_pair=False,  # momentum encoder for encoder and projection
        tau=0.005,  # note MoCo uses 0.001 instead
        # see https://github.com/facebookresearch/moco/blob/78b69cafae80bc74cd1a89ac3fb365dc20d157d3/moco/builder.py#L15
        # and https://github.com/facebookresearch/moco/blob/78b69cafae80bc74cd1a89ac3fb365dc20d157d3/moco/builder.py#L50
        batch_as_neg=True,
    ):
        super().__init__()
        assert len(obs_space.shape) == 3  # check if image space
        self.obs_space = obs_space
        self.embedding_dim = embedding_dim
        self.optimize_interval = optimize_interval
        self.detach_embedding = detach_embedding

        self.pair_type = pair_type
        self.similarity = similarity
        self.temperature = temperature
        self.use_projection_head = use_projection_head
        self.normalize_projected = normalize_projected
        self.normalize_logits = normalize_logits
        self.detach_projected_pair = detach_projected_pair
        self.aug_pair_only = aug_pair_only
        self.detach_augmented = detach_augmented
        self.use_target_for_pair = use_target_for_pair
        self.tau = tau
        self.batch_as_neg = batch_as_neg

        in_channels = self.obs_space.shape[0]
        img_size = (self.obs_space.shape[1], self.obs_space.shape[2])

        self.aug = builder.build_aug(aug_params, img_size=img_size)

        self.encoder = builder.build_network_modules(
            encoder_network_params, in_channels=in_channels
        )
        self.encoder = nn.Sequential(*self.encoder)

        self.conv_output_shape, n_flatten = compute_output_shape(
            self.obs_space.sample(), self.encoder, aug=self.aug
        )

        self.fc_encoder = nn.Linear(n_flatten, self.embedding_dim)

        if self.use_projection_head:
            # only used for the contrastive objective, see see SimCLR 4.2 of https://arxiv.org/pdf/2002.05709.pdf#page=6
            # MoCo has encoder for query and momentum encoder for key
            # see https://github.com/facebookresearch/moco/blob/90e244a23d135e8facca41f8aa47541cc30c61b2/moco/builder.py#L144
            # MoCo2 follows SimCLR w/ projection head MLP (just 2 linear layers)
            # see SimCLR https://arxiv.org/pdf/2002.05709.pdf#page=2 and MoCo2 https://arxiv.org/pdf/2003.04297.pdf#page=2

            if projection_head_layers == 2:
                self.projection_head = nn.Sequential(
                    *[
                        nn.Linear(embedding_dim, embedding_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(embedding_dim, embedding_dim),
                    ]
                )
            elif projection_head_layers == 3:
                self.projection_head = nn.Sequential(
                    *[
                        nn.Linear(embedding_dim, embedding_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(embedding_dim, embedding_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(embedding_dim, embedding_dim),
                    ]
                )
            else:
                raise ValueError
        else:
            self.projection_head = None

        if self.similarity == 'bilinear':
            self.W = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            # self.W = nn.Parameter(torch.rand(self.embedding_dim, self.embedding_dim)) # different weight init

        self.optimizer = builder.build_optim(optim_params, params=self.parameters())

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.apply(module_reset_parameters)
        self.fc_encoder.reset_parameters()
        if self.use_projection_head:
            self.projection_head.apply(module_reset_parameters)
        if self.similarity == 'bilinear':
            self.W.reset_parameters()

        self._create_target_networks()

    def _create_target_networks(self):
        if self.use_target_for_pair:
            self.encoder_target = copy.deepcopy(self.encoder)
            self.encoder_target.load_state_dict(self.encoder.state_dict())

            self.fc_encoder_target = copy.deepcopy(self.fc_encoder)
            self.fc_encoder_target.load_state_dict(self.fc_encoder.state_dict())

            if self.use_projection_head:
                self.projection_head_target = copy.deepcopy(self.projection_head)
                self.projection_head_target.load_state_dict(self.projection_head.state_dict())

    def forward(self, x, with_conv_output=False, detach_embedding=None):
        z = self.encode(x)

        detach_embedding = (
            detach_embedding if detach_embedding is not None else self.detach_embedding
        )
        if detach_embedding:
            z = z.detach()

        if with_conv_output:
            raise NotImplementedError
        else:
            return z

    def encode(self, x):
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        z = self.fc_encoder(h)
        return z

    def _encode_target(self, x):
        h = self.encoder_target(x)
        h = h.flatten(start_dim=1)
        z = self.fc_encoder_target(h)
        return z

    def contrast(self, anchor, pair, pairwise=True):
        # alternatively known as query and key
        # if pairwise=True, compares all pairs in batch, otherwise gives elementwise pair similarity

        # NOTE: ATC uses an MLP only for the anchor which is
        # similar to the prediction head for BYOL, SimSiam
        # while SimCLR and MoCo apply symmetrically to anchor and pair
        if self.use_projection_head:
            anchor = self.projection_head(anchor)

            if self.use_target_for_pair:
                # as https://github.com/astooke/rlpyt/blob/b05f954e88fc774d61c6504ebe62ff71a181ad7a/rlpyt/ul/algos/ul_for_rl/augmented_temporal_contrast.py#L141
                assert self.detach_projected_pair
                with torch.no_grad():
                    pair = self.projection_head_target(anchor)
            else:
                pair = self.projection_head(pair)

                # this also means pair will not update embedding_head.encoder
                if self.detach_projected_pair:
                    pair = pair.detach()

        # MoCo and SimCLR use cosine similarity and normalize embeddings by l2
        # see section 5.1 of SimCLR, https://arxiv.org/pdf/2002.05709.pdf#page=5
        # also https://github.com/facebookresearch/moco/blob/90e244a23d135e8facca41f8aa47541cc30c61b2/moco/builder.py#L126
        # and https://github.com/google-research/simclr/blob/ac4626a6cc7ecfd4bd95bcccdb2fbe646028a4f7/objective.py#L55
        if self.normalize_projected:  # if used w/ dotproduct, then cosine similarity
            anchor = F.normalize(anchor, dim=1)
            pair = F.normalize(pair, dim=1)

        # NOTE: pairwise=False should just give the main diagonal when pairwise=True
        # for scores, anchor corresponds to row, pair corresponds to col
        if self.similarity == 'bilinear':
            if pairwise:
                # as https://github.com/astooke/rlpyt/blob/b05f954e88fc774d61c6504ebe62ff71a181ad7a/rlpyt/ul/models/ul/atc_models.py#L27
                logits = torch.matmul(self.W(anchor), pair.t())
                # logits = torch.matmul(anchor @ self.W.weight.t(), pair.t())
            else:
                # as https://github.com/thanard/hallucinative-topological-memory/blob/82f63f01e7b6b552d515275249d5a11a5be6fe0a/models.py#L299
                # z1 = z1.unsqueeze(2) # bs x z_dim x 1
                # z2 = z2.unsqueeze(2)
                # w = self.W
                # w = w.repeat(z1.size(0), 1, 1)
                # f_out = torch.bmm(torch.bmm(z2.permute(0, 2, 1), w), z1)
                # f_out = f_out.squeeze(-1)

                # (bs x 1 x z_dim) @ (bs x z_dim x z_dim) @ (bs x z_dim x 1) == (bs x 1 x 1)
                logits = (pair.unsqueeze(1) @ self.W.weight) @ anchor.unsqueeze(2)
                logits = logits.squeeze(-1)
        elif self.similarity == 'dotproduct':
            if pairwise:
                logits = anchor @ pair.t()
                # https://github.com/facebookresearch/moco/blob/90e244a23d135e8facca41f8aa47541cc30c61b2/moco/builder.py#L141
                # l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
                # l_pos = (q * k).sum(dim=1, keepdim=True)
                # l_pos = (q.unsqueeze(1) @ k.unsqueeze(2)).squeeze(2)
                #
                # l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
                # l_neg = q @ k_neg.t() # queue stores as (z, B), so transpose
            else:
                logits = pair.unsqueeze(1) @ anchor.unsqueeze(2)
                logits = logits.squeeze(-1)
        elif self.similarity == 'negdist':
            if pairwise:
                # as CFM
                # see https://github.com/wilson1yan/contrastive-forward-model/blob/465a2a5a1eb5c95abe4d2605db02e5e9ef1b485c/cfm/train_cfm.py#L41
                # https://github.com/wilson1yan/contrastive-forward-model/blob/465a2a5a1eb5c95abe4d2605db02e5e9ef1b485c/cfm/evaluate_planning.py#L106
                logits = -torch.cdist(anchor, pair, p=2)
            else:
                logits = -torch.norm(anchor - pair, p=2, dim=1, keepdim=True)
        else:
            raise ValueError(f'Unsupported similarity: {self.similarity}')

        logits /= self.temperature
        return logits

    def optimize(self, x, embedding_target=None):
        if self.pair_type == 'state':
            assert embedding_target is None
            if self.detach_augmented:
                embedding_target = x.detach().clone()
            else:
                embedding_target = x
        elif self.pair_type == 'next_state':
            assert embedding_target is not None
        elif self.pair_type == 'sampled_pair':
            assert embedding_target is not None
            state_pair, distance_pair, pos_mask = embedding_target

            if self.batch_as_neg:
                B = x.size(0)
                assert pos_mask.long().sum() == B  # must all be positive examples
            embedding_target = state_pair
        else:
            raise ValueError

        # augmentations should be different b/w views and elements in batch
        if self.aug is not None:
            if self.detach_augmented:
                with torch.no_grad():
                    if not self.aug_pair_only:
                        x = self.aug(x)
                    embedding_target = self.aug(embedding_target)
            else:
                if not self.aug_pair_only:
                    x = self.aug(x)
                embedding_target = self.aug(embedding_target)

        a_z = self.encode(x)
        if self.use_target_for_pair:
            assert self.detach_projected_pair
            with torch.no_grad():
                p_z = self._encode_target(embedding_target)
        else:
            p_z = self.encode(embedding_target)

        if self.batch_as_neg:
            # use other elements in in batch as negatives
            opt_info = self._optimize_batch_neg(a_z, p_z)
        else:
            assert self.pair_type == 'sampled_pair'
            # use explicit negatives sampled per element in batch
            opt_info = self._optimize_explicit_neg(a_z, p_z, pos_mask)

        if self.use_target_for_pair:
            update_target_network(self.encoder, self.encoder_target, tau=self.tau)
            update_target_network(self.fc_encoder, self.fc_encoder_target, tau=self.tau)
            update_target_network(self.projection_head, self.projection_head_target, tau=self.tau)
        return opt_info

    def _optimize_batch_neg(self, anchor, positive):
        opt_info = {}

        logits = self.contrast(anchor, positive, pairwise=True)
        scores = logits.detach()

        # use byte b/c cannot invert bool in torch1.1.0
        pos_mask = torch.zeros_like(logits, dtype=torch.uint8)
        # fill_diagonal_ not available in torch1.1.0
        pos_mask.diagonal().fill_(1)
        # pos_mask = torch.eye(logits.shape[-1], device=logits.device, dtype=torch.uint8) # use byte b/c cannot call eye for bool
        neg_mask = ~pos_mask

        opt_info['contrast_similarity_pos'] = untorchify(scores[pos_mask].mean(dim=0))
        opt_info['contrast_similarity_neg'] = untorchify(scores[neg_mask].mean(dim=0))

        # normalize to range (-inf, 0]
        # https://github.com/astooke/rlpyt/blob/b05f954e88fc774d61c6504ebe62ff71a181ad7a/rlpyt/ul/models/ul/atc_models.py#L29
        # https://github.com/astooke/rlpyt/blob/b05f954e88fc774d61c6504ebe62ff71a181ad7a/rlpyt/ul/algos/ul_for_rl/cpc.py#L192
        #
        # https://github.com/thanard/hallucinative-topological-memory/blob/82f63f01e7b6b552d515275249d5a11a5be6fe0a/models.py#L26
        # https://github.com/KevinMusgrave/pytorch-metric-learning/blob/0b575b556fe339c2a62043d0ff0efe7fe85107bc/src/pytorch_metric_learning/losses/ntxent_loss.py#L29
        if self.normalize_logits:
            logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()
            # detach for numerical stability
            # https://github.com/KevinMusgrave/pytorch-metric-learning/issues/282

        # labels are indices for positive keys
        # could also pass identity matrix?
        labels = torch.arange(logits.shape[0], dtype=torch.long, device=logits.device)
        nce_loss = F.cross_entropy(logits, labels)

        self.optimizer.zero_grad()
        nce_loss.backward()
        self.optimizer.step()

        opt_info['embedding_head_loss'] = nce_loss.item()
        correct = (torch.argmax(scores, dim=1) == labels).float()
        opt_info['contrast_accuracy'] = correct.mean().item()

        return opt_info

    def _optimize_explicit_neg(self, anchor, pair, pos_mask):
        opt_info = {}

        y = pos_mask.long()
        B = anchor.size(0)
        assert y[:B].sum() == B
        assert y[B:].sum() == 0
        positive = pair[:B, ...]  # B x z
        negative = pair[B:, ...]  # (B x K) x z, where K is the number of neg examples
        K = negative.size(0) // anchor.size(0)

        # Suppose we have B=3 and K=2, then
        # negative == [00, 10, 20, 01, 11, 21] (where 01 correponds to batch
        # element 0 and negative example 1), and similarly
        # for negative_log_density. We want to reshape get [[00, 01], [10, 11], [20, 21]]
        # so the negatives are paired with the right batch element.

        positive_log_density = self.contrast(anchor, positive, pairwise=False)
        _negative_log_density = self.contrast(
            anchor.repeat(K, 1), negative, pairwise=False
        )  # use repeat to match np.tile, https://pytorch.org/docs/stable/tensors.html#torch.Tensor.repeat
        negative_log_density = _negative_log_density.view(K, B).transpose(0, 1)
        # if we do .view(B, K) then we get [[00, 10], [20, 01], [11, 21]]

        # logits: Nx(1+K)
        logits = torch.cat([positive_log_density, negative_log_density], dim=1)

        if self.normalize_logits:
            logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()
            # detach for numerical stability
            # https://github.com/KevinMusgrave/pytorch-metric-learning/issues/282

        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        nce_loss = F.cross_entropy(logits, labels)

        # TODO: also use other anchor elements as negatives?

        # related ref: https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/losses/ntxent_loss.py#L29

        # _p_density_ratio = torch.zeros(
        #     (B, 1), dtype=positive_log_density.dtype, device=positive_log_density.device
        # )
        # density_ratio = torch.cat(
        #     [_p_density_ratio, negative_log_density - positive_log_density], dim=1
        # )
        # # # https://github.com/thanard/hallucinative-topological-memory/blob/82f63f01e7b6b552d515275249d5a11a5be6fe0a/models.py#L58
        # # density_ratio = torch.cat([density_ratio, -positive_log_density, negative_log_density], dim=1)
        # assert self.normalize_logits
        # max_d = torch.max(density_ratio, dim=1, keepdim=True)[0].detach()
        # log_sum_exp = max_d + torch.log(torch.sum(torch.exp(density_ratio - max_d), dim=1))
        # nce_loss = -torch.mean(log_sum_exp)

        self.optimizer.zero_grad()
        nce_loss.backward()
        self.optimizer.step()

        opt_info['embedding_head_loss'] = nce_loss.item()
        opt_info['contrast_scores_pos'] = untorchify(
            torch.sigmoid(positive_log_density.mean(dim=0))
        )
        opt_info['contrast_scores_neg'] = untorchify(
            torch.sigmoid(_negative_log_density.mean(dim=0))
        )
        opt_info['contrast_accuracy'] = None  # TODO

        return opt_info

    @property
    def opt_info_keys(self):
        return [
            'embedding_head_loss',
            'contrast_accuracy',
            'contrast_similarity_pos',
            'contrast_similarity_neg',
        ]

    def state_dict(self, *args, **kwargs):
        state_dict = dict(
            model=super().state_dict(*args, **kwargs),
            optimizer=self.optimizer.state_dict(),
        )
        return state_dict

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict.pop('optimizer'))
        super().load_state_dict(state_dict['model'])
