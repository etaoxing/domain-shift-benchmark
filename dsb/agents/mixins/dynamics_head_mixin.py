from dsb.dependencies import *


class DynamicsHeadMixin:
    def predict_dynamics_action(
        self, state, next_state, state_embedding=None, next_state_embedding=None
    ):
        if self.embedding_head is not None:
            state = self.compute_embedding(state, state_embedding=state_embedding)
            next_state = self.compute_embedding(next_state, state_embedding=next_state_embedding)
        else:
            raise NotImplementedError

        pred_action = self.dynamics_head.inverse_model(state, next_state)
        return pred_action

    def predict_dynamics_obs(self, state, state_embedding=None):
        if self.embedding_head is not None:
            state = self.compute_embedding(state, state_embedding=state_embedding)
        else:
            raise NotImplementedError

        pred_obs = self.dynamics_head.obs_model(state['achieved_goal'])
        return pred_obs

    def optimize_dynamics_head(
        self, buffer, ptrs, batch, state_embedding=None, next_state_embedding=None
    ):
        if self.optimize_iterations % self.dynamics_head.optimize_interval == 0:
            _state, action, _next_state, reward, done = batch

            if self.embedding_head is not None:
                state = self.compute_embedding(_state, state_embedding=state_embedding)
                next_state = self.compute_embedding(
                    _next_state, state_embedding=next_state_embedding
                )
            else:
                raise NotImplementedError

            opt_info = self.dynamics_head.optimize(state, next_state, action)
        else:
            opt_info = {k: None for k in self.dynamics_head.opt_info_keys}
        return opt_info
