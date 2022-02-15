class BufferWrapper:
    def __init__(self, buffer, **unused):
        self.buffer = buffer

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __getattr__(self, attr):
        return getattr(self.buffer, attr)

    def __repr__(self):
        b = (
            str(self.buffer)
            if isinstance(self.buffer, BufferWrapper)
            else f' ({type(self.buffer).__name__})'
        )
        return f'<{type(self).__name__}{b}>'
