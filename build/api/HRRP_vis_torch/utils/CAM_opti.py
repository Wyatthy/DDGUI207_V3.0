import cv2
import numpy as np
import torch
from typing import Any, List, Tuple
from utils.CAM import BaseCAM, GradCAM, GradCAMpp, XGradCAM, EigenGradCAM, LayerCAM, t2n    


class CAMbasedDropout():
    def __init__(self, dropRate=0.1, maxFeatureStride=32) -> None:
        self._reset_()
        self.maxFeatureStride = maxFeatureStride
        self.dropRate = dropRate 

    def __call__(self, featureMaps, CAMs, featureStride) -> Any:
        self.featureMaps = featureMaps
        self.CAMs = CAMs
        
        self.featureStride = featureStride
        self.bz, self.nc, self.h, self.w = featureMaps.shape

        self.dropMasks = self._compute_mask_()

        return self.featureMaps*self.dropMasks[:, None, :, :]

    
    def _compute_mask_(self):
        # for every data in a batch
        dropMasks = torch.ones((self.bz, self.h, self.w), \
                                    device=self.featureMaps.device)
        for featureMap, CAM, dropMask in zip(self.featureMaps, self.CAMs, \
                                             dropMasks):
            # 小于dropRate的概率，不drop
            # if torch.rand(1, device=self.featureMaps.device) < self.dropRate:
            #     continue
            # [x] compute how many activation units to drop
            featureSize = featureMap.shape[1:]
            dropBlockWidth = (self.featureStride*featureSize[0])//(self.maxFeatureStride*2)
            dropBlockHeight = (self.featureStride*featureSize[1])//(self.maxFeatureStride*2)

            dropSize = [dropBlockWidth, 1]
            keepSize = [(featureSize[0]-dropSize[0]+1),
                                (featureSize[1]-dropSize[1]+1)]
            
            # [x] compute how many dropblocks -> k
            gamma = self._compute_gamma(featureSize, dropSize)
            gamma = max(gamma, 0)
            gamma = min(gamma, 1)
            k = torch.sum(torch.bernoulli(torch.full(keepSize, gamma,\
                dtype=torch.float, device=self.featureMaps.device))).item()
            k = min(k, featureSize[0]*featureSize[1]-1)

            # [x] compute topk positions of CAM in bbox
            topkValue, indices = torch.flatten(torch.from_numpy(CAM))\
                            .topk(int(k), largest=True, sorted=True)
            topkPositions = np.array([[indice//featureSize[1], 
                            int(indice%featureSize[1])] for indice in indices])

            # [x] compute drop positions in featureMap
            for seed in topkPositions:
                dropMask[seed[0]-dropSize[0]//2:seed[0]+dropSize[0]//2+1
                    , seed[1]-dropSize[1]//2:seed[1]+dropSize[1]//2+1]=0
        return dropMasks


    def _reset_(self):
        '''
            Release all variables
        '''
        self.featureMaps = None
        self.CAMs = None

        self.featureStride = None
        self.bz, self.nc, self.h, self.w = None, None, None, None
        self.dropMasks = None

    '''
    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature map on which some areas will be randomly
                dropped.

        Returns:
            Tensor: The tensor after DropBlock layer.
        """
        if not self.training:
            return x
        self.iter_cnt += 1
        N, C, H, W = list(x.shape)
        gamma = self._compute_gamma((H, W))
        mask_shape = (N, C, H - self.blockSize + 1, W - self.blockSize + 1)
        mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))

        mask = F.pad(mask, [self.blockSize // 2] * 4, value=0)
        mask = F.max_pool2d(
            input=mask,
            stride=(1, 1),
            kernel_size=(self.blockSize, self.blockSize),
            padding=self.blockSize // 2)
        mask = 1 - mask
        x = x * mask * mask.numel() / (self.eps + mask.sum())
        return x
    '''

    def _compute_gamma(self, featSize, blockSize):
        """Compute the value of gamma according to paper. gamma is the
        parameter of bernoulli distribution, which controls the number of
        features to drop.

        gamma = (drop_prob * fm_area) / (drop_area * keep_area)

        Args:
            featSize (tuple[int, int]): The height and width of feature map.
            blockSize (int): the size of dropout block.

        Returns:
            float: The value of gamma.
        """
        gamma = (self.dropRate * featSize[0] * featSize[1]) 
        gamma /= ((featSize[0] - blockSize[0] + 1) *
                  (featSize[1] - blockSize[1] + 1)) + 1e-7
        gamma /= (blockSize[0]*blockSize[1])
        return gamma