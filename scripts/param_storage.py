
class ParamStorage:
    def __init__(self):
        self.params = []
        self.frozen_indices = []

    def set_params(self, params):
        temp = []
        for i, param in enumerate(params):
            if i not in self.frozen_indices:
                temp.append(param)
            else:
                temp.append(self.params[i])
        self.params = temp

    def get_params(self):
        return self.params.copy()

    def freeze(self, indices=None):
        if len(self.params) == 0:
            raise ValueError('No parameters to freeze')

        if indices is None:
            indices = list(range(len(self.get_params())))
        self.frozen_indices = indices
