from dsb.dependencies import *


class SiameseGCEmbeddingHeadWrapper(nn.Module):
    # image encoder for obs and goal are shared
    # only the obs images are used to update the encoder
    def __init__(self, head):
        super().__init__()
        self.head = head

    def __getattr__(self, attr):
        try:
            a = super().__getattr__(attr)
        except:
            a = getattr(self.head, attr)
        return a

    def forward(self, x, inplace=True, with_conv_output=False, detach_embedding=None, **kwargs):
        y = x if inplace else dict()

        for k in [
            'achieved_goal',
            'fs_achieved_goal',
            'c_achieved_goal',
            'desired_goal',
            'fs_desired_goal',
            'c_desired_goal',
        ]:
            # current observation, framestack of previous frames, and context (previous subgoal or episode start)
            if k not in x.keys():
                continue

            de = detach_embedding
            if k.startswith('c_'):  # always detaching context
                de = True

            if k.startswith('fs_'):
                k_ = k[3:]
                x_ = x[k]

                x_s = tuple(x_.shape)
                B, T = x_s[0], x_s[1]
                o = self.head(
                    x_.view((B * T,) + x_s[2:]),
                    with_conv_output=with_conv_output,
                    detach_embedding=True if (k_ == 'desired_goal' and de is None) else de,
                    **kwargs,
                )

                # concatenating tuple with torch.Size should cast to tuple
                if with_conv_output:
                    z, co = o
                    y[k] = z.reshape((B, T) + z.shape[1:])
                    y[f'{k}_conv_output'] = co.reshape((B, T) + co.shape[1:])
                else:
                    z = o
                    y[k] = o.reshape((B, T) + z.shape[1:])
            else:
                o = self.head(
                    x[k],
                    with_conv_output=with_conv_output,
                    detach_embedding=True if (k == 'desired_goal' and de is None) else de,
                    **kwargs,
                )
                if with_conv_output:
                    y[k], y[f'{k}_conv_output'] = o
                else:
                    y[k] = o

            # NOTE: by default, always detaching desired_goal embedding
            # if not detaching, then may need to retain_graph in downstream losses
            # RuntimeError: Trying to backward through the graph a second time, but the saved intermediate results have already been freed.
        return y

    def optimize(self, x, **kwargs):
        return self.head.optimize(x['achieved_goal'], **kwargs)
