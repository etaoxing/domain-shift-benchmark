from dsb.dependencies import *


class LinearlyDecayingScheduler(nn.Module):
    def __init__(
        self,
        initial_p=1.0,
        final_p=0.001,
        schedule_iterations=1000,
        delay_iterations=0,  # set equal to initial_collect_steps
    ):
        super().__init__()
        self.initial_p = initial_p
        self.final_p = final_p
        self.schedule_iterations = schedule_iterations
        self.delay_iterations = delay_iterations

    def step(self, iteration):
        assert iteration >= 0
        it = max(0, float(iteration - self.delay_iterations))
        progress = min(it / (self.schedule_iterations), 1)
        if progress == 1:
            return self.final_p
        return self.initial_p + progress * (self.final_p - self.initial_p)
