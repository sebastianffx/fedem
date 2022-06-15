import torch
from torch.nn.modules.loss import _Loss
from typing import Callable, List, Optional, Sequence, Union

class BlobLoss(_Loss):
    """
    Compute Blob loss between two tensors

    The original paper: F. Kofler et al., “blob loss: instance imbalance aware loss functions for semantic segmentation.”. Available: http://arxiv.org/abs/2205.08209

    """
    def __init__(
        self,
        loss_function : Callable,
        lambda_main: float = 0,
        lambda_blob: float = 1,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        batch: bool = False,
    ) -> None:
        """
        Args:
            loss_function: callable loss function
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.

        Raises:
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.loss_function = loss_function
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.batch = batch

        self.lambda_main = lambda_main
        self.lambda_blob = lambda_blob

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        if self.lambda_main > 0:
            #the reduction should happen later
            self.loss_function(input, target, reduction=LossReduction.NONE)
        else:
            main_loss = 0

        if self.lambda_blob > 0:
            blob_loss = compute_compound_loss(input, target, reduce_axis=reduce_axis)
        else:
            blob_loss = 0
        
        f: torch.Tensor = self.lambda_main*main_loss + self.lambda_blob*blob_loss

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f

    def compute_compound_loss(input: torch.Tensor, target: torch.Tensor):
        """ Compute blob loss for a batch of samples
        """
        #TODO: try to vectorize the entire thing, might be difficult due to the varying number of blob
        #UNLESS we add zero for the labels that other samples don't have in their labels!
        """
        #it does not make sense to reduce along the batch or channel, would merge the labels
        unique_labels = torch.unique(target, dim=torch.arange(2, len(input.shape)).tolist()) #.shape = batch, channel, 1
        then iterate over the sample in the batch?
        """

        samples_blob_loss = []
        # loop over elements in the batch
        for sample_idx in range(input.shape[0]):
            #extract label of this specific sample
            if sample_idx < input.shape[0]:
                sample_label = target[sample_idx:sample_idx+1, ...]
                sample_pred = input[sample_idx:sample_idx+1, ...]
            else:
                sample_label = target[sample_idx:None, ...]
                sample_pred = input[sample_idx:None, ...]

            # each component/blob correspond to a unique label
            unique_labels = torch.unique(sample_label, sorted=True)
            if unique_labels[0]==0:
                unique_labels = unique_labels[1:]

            sample_loss = []
            for ula in unique_labels:
                #extract all the positive labels
                label_mask = sample_label > 0
                # we flip labels, so that the background become visible -> all blob are set to False
                label_mask = ~label_mask
                # we set the mask to true where our label of interest is located, resulting mask = everything but the other blobs
                label_mask[sample_label == ula] = 1
                #hide the target for everything but the blob considered
                blob_label = sample_label == ula
                #hide the prediction of the other blobs
                masked_pred = sample_pred*label_mask
                try:
                    # we try with int labels first, but some losses require floats
                    blob_loss = self.loss_function(masked_pred, blob_label.int(), reduction=LossReduction.NONE)
                except:
                    # if int does not work we try float
                    blob_loss = self.loss_function(masked_pred, blob_label.float(), reduction=LossReduction.NONE)

                sample_loss.append(blob_loss)

            #normalize blob loss by the number of blob -> hope that it does not break autograd
            samples_blob_loss.append(torch.average(torch.stack(sample_loss), dim=-1))
        
        #stack them so that the batch dimension is restaured
        output = torch.stack(samples_blob_loss, dim=0)

        #input and output should have the same shape
        print(output.shape)
        print(input.shape)
        assert (input.shape == output.shape)

        return output