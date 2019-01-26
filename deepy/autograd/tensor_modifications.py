import numpy as np

from deepy.autograd import Autograd


class Reshape(Autograd):
    def __init__(self, *new_shape):
        self.new_shape = new_shape

    def forward(self, ctx: Autograd.Context, tensor: np.array):
        ctx.save_for_back(tensor.shape)
        return np.reshape(tensor, self.new_shape)

    def backward(self, ctx: Autograd.Context, grad: np.array):
        old_shape = ctx.data_for_back
        return np.reshape(grad, old_shape)


class SwapAxes(Autograd):
    def __init__(self, axis1, axis2):
        self.axis1 = axis1
        self.axis2 = axis2

    def forward(self, ctx: Autograd.Context, tensor):
        return np.swapaxes(tensor, self.axis1, self.axis2)

    def backward(self, ctx: Autograd.Context, grad: np.array = None):
        return np.swapaxes(grad, self.axis1, self.axis2)


class GetItem(Autograd):
    def __init__(self, item):
        self.item = item

    def forward(self, ctx: Autograd.Context, tensor):
        ctx.save_for_back(tensor.shape)
        return tensor[self.item]

    def backward(self, ctx: Autograd.Context, grad: np.array):
        old_shape = ctx.data_for_back
        new_grad = np.zeros(old_shape, dtype=np.float32)
        new_grad[self.item] = grad
        return new_grad


class Img2Col(Autograd):
    def __init__(self, kernel_size, stride=1):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, ctx: Autograd.Context, image: np.array):
        """
        Performs Img to Col transformation.
        Args:
            ctx (Autograd.Context): usual context,
            image (np.array):   image to be transformed, allowed shapes:
                                [N, C, H, W], [C, H, W]
                                N - batches,
                                C - channels,
                                H - height,
                                W - width
        """

        has_batches = len(image.shape) == 4

        img_w = image.shape[-1]
        img_h = image.shape[-2]
        channels = image.shape[-3]

        # new image width
        new_w = (img_w - self.kernel_size[0]) // self.stride[0] + 1

        # new image height
        new_h = (img_h - self.kernel_size[1]) // self.stride[1] + 1

        ctx.save_for_back(image.shape, channels, new_w)

        flattened_part_shape = (image.shape[0], -1) if has_batches else (-1)

        ret_shape = (channels * self.kernel_size[0] * self.kernel_size[1], new_w * new_h)
        if has_batches:
            ret_shape = (image.shape[0], *ret_shape)

        ret_image = np.zeros(ret_shape)
        for i in range(new_h):
            for j in range(new_w):
                part = image[
                       ...,
                       i * self.stride[1]:i * self.stride[1] + self.kernel_size[1],
                       j * self.stride[0]:j * self.stride[0] + self.kernel_size[0]]

                part = np.reshape(part, flattened_part_shape)
                ret_image[..., :, i * new_w + j] = part

        return ret_image

    def backward(self, ctx: Autograd.Context, grad: np.array = None):
        old_shape, channels, old_w = ctx.data_for_back

        ret_grad = np.zeros(old_shape, dtype=np.float32)

        for i in range(grad.shape[-1]):
            col = grad[..., :, i]
            col = np.reshape(
                col,
                (-1, channels, self.kernel_size[1], self.kernel_size[0])
            )
            h_start = (i // old_w) * self.stride[1]
            w_start = (i % old_w) * self.stride[0]

            ret_grad[..., h_start:h_start + self.kernel_size[1], w_start:w_start + self.kernel_size[0]] += col

        return ret_grad
