class BasePolicy:
    def __init__(self, agent, **unused):
        self.agent = agent

    def select_action(self, state, deterministic=False, **kwargs):
        return self.agent.select_action(state, deterministic=deterministic, **kwargs)

    def get_stats(self):
        stats = self.agent.get_stats()
        return stats

    def reset_stats(self):
        self.agent.reset_stats()

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __getattr__(self, attr):
        return getattr(self.agent, attr)

    def __repr__(self):
        a = str(self.agent) if isinstance(self.agent, BasePolicy) else ' (agent)'
        return f'<{type(self).__name__}{a}>'
