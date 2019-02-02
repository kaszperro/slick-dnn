from typing import Tuple, Union

import numpy as np

from deepy.autograd import Autograd, Context


class Reshape(Autograd):
    def __init__(self, *new_shape):
        self.new_shape = new_shape

    def forward(self, ctx: Context, tensor: np.array):
        ctx.save_for_back(tensor.shape)
        return np.reshape(tensor, self.new_shape)

    def backward(self, ctx: Context, grad: np.array):
        old_shape = ctx.data_for_back
        return np.reshape(grad, (grad.shape[0], *old_shape[1:]))


class SwapAxes(Autograd):
    def __init__(self, axis1, axis2):
        self.axis1 = axis1
        self.axis2 = axis2

    def forward(self, ctx: Context, tensor):
        return np.swapaxes(tensor, self.axis1, self.axis2)

    def backward(self, ctx: Context, grad: np.array = None):
        return np.swapaxes(grad, self.axis1, self.axis2)


class GetItem(Autograd):
    def __init__(self, item):
        self.item = item

    def forward(self, ctx: Context, tensor):
        ctx.save_for_back(tensor.shape)
        return tensor[self.item]

    def backward(self, ctx: Context, grad: np.array):
        old_shape = ctx.data_for_back
        new_grad = np.zeros(old_shape, dtype=np.float32)
        new_grad[self.item] = grad
        return new_grad


class Img2Col(Autograd):
    @staticmethod
    def img_2_col_forward(kernel_size, stride, merge_channels, image):
        has_batches = len(image.shape) == 4

        img_w = image.shape[-1]
        img_h = image.shape[-2]
        channels = image.shape[-3]

        # new image width
        new_w = (img_w - kernel_size[0]) // stride[0] + 1

        # new image height
        new_h = (img_h - kernel_size[1]) // stride[1] + 1

        if merge_channels:
            ret_shape = (channels * kernel_size[0] * kernel_size[1], new_w * new_h)
            flattened_part_shape = (-1,)
        else:
            ret_shape = (channels, kernel_size[0] * kernel_size[1], new_w * new_h)
            flattened_part_shape = (channels, -1)

        if has_batches:
            ret_shape = (image.shape[0], *ret_shape)
            flattened_part_shape = (image.shape[0], *flattened_part_shape)

        ret_image = np.zeros(ret_shape)
        for i in range(new_h):
            for j in range(new_w):
                part = image[
                       ...,
                       i * stride[1]:i * stride[1] + kernel_size[1],
                       j * stride[0]:j * stride[0] + kernel_size[0]]

                part = np.reshape(part, flattened_part_shape)
                ret_image[..., :, i * new_w + j] = part

        return ret_image

    @staticmethod
    def img_2_col_backwards(kernel_size, stride, old_shape, grad, mask=None):
        channels = old_shape[-3]

        img_w = old_shape[-1]

        # new image width
        old_w = (img_w - kernel_size[0]) // stride[0] + 1

        ret_grad = np.zeros(old_shape, dtype=np.float32)

        for i in range(grad.shape[-1]):
            col = grad[..., :, i]
            col = np.reshape(
                col,
                (-1, channels, kernel_size[1], kernel_size[0])
            )

            if mask is not None:
                col_mask = mask[..., :, i]
                col_mask = np.reshape(col_mask, (-1, channels, kernel_size[1], kernel_size[0]))
                col *= col_mask

            h_start = (i // old_w) * stride[1]
            w_start = (i % old_w) * stride[0]

            ret_grad[..., h_start:h_start + kernel_size[1], w_start:w_start + kernel_size[0]] += col

        return ret_grad

    def __init__(self, kernel_size, stride: Union[int, Tuple[int, int]] = 1):

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, ctx: Context, image: np.array):
        """
        Performs Img to Col transformation.
        Args:
            ctx (Context): usual context,
            image (np.array):   image to be transformed, allowed shapes:
                                [N, C, H, W], [C, H, W]
                                N - batches,
                                C - channels,
                                H - height,
                                W - width
        """

        ctx.save_for_back(image.shape)

        return self.img_2_col_forward(
            self.kernel_size,
            self.stride,
            True,
            image
        )

    def backward(self, ctx: Context, grad: np.array = None):
        old_shape = ctx.data_for_back

        return self.img_2_col_backwards(
            self.kernel_size,
            self.stride,
            old_shape,
            grad,
            None
        )


class MaxPool2d(Autograd):
    def __init__(self, kernel_size, stride=1):

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, ctx: Context, image):
        """
        Performs 2d max pool over input tensor

        Args:
            ctx (Context): Autograd Conext
            image (np.array):  input image. Allowed shapes:
                                [N, C, H, W], [C, H, W]
                                N - batches,
                                C - channels,
                                H - height,
                                W - width

       Returns:
           tensor (np.array):

        """
        img_w = image.shape[-1]
        img_h = image.shape[-2]
        channels = image.shape[-3]

        # new image width
        new_w = (img_w - self.kernel_size[0]) // self.stride[0] + 1

        # new image height
        new_h = (img_h - self.kernel_size[1]) // self.stride[1] + 1

        img_out = Img2Col.img_2_col_forward(
            self.kernel_size,
            self.stride,
            False,
            image
        )

        maxed = np.max(img_out, -2)
        ctx.save_for_back(img_out, image.shape, maxed.shape)

        return np.reshape(maxed, (-1, channels, new_h, new_w))

    def backward(self, ctx: Context, grad: np.array = None):
        reshaped_image, old_shape, maxed_shape = ctx.data_for_back
        grad = np.reshape(grad, maxed_shape)

        mask = (reshaped_image == np.max(reshaped_image, -2, keepdims=True))

        new_grad = np.zeros_like(reshaped_image)
        for i in range(grad.shape[-1]):
            grad_col = np.reshape(grad[..., :, i], -1)
            new_grad[..., :, i][mask[..., :, i]] = grad_col

        return Img2Col.img_2_col_backwards(
            self.kernel_size,
            self.stride,
            old_shape,
            new_grad,
            None
        )
