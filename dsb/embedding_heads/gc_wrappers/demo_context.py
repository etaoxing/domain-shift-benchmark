from dsb.dependencies import *


class GCContextDemoEmbeddingHeadWrapper(nn.Module):
    def __init__(self, head, hidden_output=False):
        super().__init__()
        self.head = head
        self.hidden_output = hidden_output

    def __getattr__(self, attr):
        try:
            a = super().__getattr__(attr)
        except:
            a = getattr(self.head, attr)
        return a

    def forward(self, x, **kwargs):
        # TODO: separate forward into pass for achieved_goal
        # and pass for desired_goal

        s = len(self.head.obs_space.shape)  # should just be 3

        _x = {}
        for k, v in x.items():
            if 'desired_goal' in k:
                pass
            else:
                # resize image obs or context demo if necessary
                if k == 'achieved_goal':
                    if len(v.shape) == s + 1:
                        # assume given time dim and add batch dim
                        v = v.unsqueeze(0)

                        # TODO: reshape batch accordingly rather than assuming
                        # all from same trajectory
                elif k == 'context_demo':
                    if len(v.shape) == s + 1:
                        # assume given time dim and add batch dim
                        # so same context across batch
                        v = v.unsqueeze(0)

                _x[k] = v

        context = _x.get('context_demo', None)

        if self.hidden_output:
            context_hidden = _x.get('context_demo_hidden', None)
            z = self.head.forward(_x, context=context, context_hidden=context_hidden, **kwargs)
            z_ag, hidden = z['achieved_goal']
            z['achieved_goal'] = z_ag
            # _x['context_demo_hidden'] = hidden
        else:
            z = self.head.forward(_x, context=context, **kwargs)
        return z

    def optimize(self, x, **kwargs):
        context = x.get('context_demo', None)
        context_hidden = x.get('context_demo_hidden', None)
        return self.head.optimize(x, context=context, context_hidden=context_hidden, **kwargs)
