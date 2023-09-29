import torch.nn as nn
class Distance(nn.Module):
    """
    Dis
    """

    def __init__(self, ):
        super(Distance, self).__init__()
        ################ Parameters #####################

    def loadmodel(self, filepath):
        print('Load all parameter from: ', filepath)

    # @autocast()
    def forward(self, features, A):
        B, C, T, MAX_N = features.shape
        print(A.shape)
        if len(A.shape) == 3:
            A = [None, ...]
        print(A.shape)
        atts = A.float()[:,:,:4,:4]
        group_scores = atts[:, :, None, :, :, None]  # .mean(dim=1)
        print(group_scores)
        return A.float