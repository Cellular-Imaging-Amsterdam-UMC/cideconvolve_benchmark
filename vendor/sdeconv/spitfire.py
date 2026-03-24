"""Spitfire variational deconvolution for 2D and 3D images."""
import torch
from ._core import SSettings
from ._interface import SDeconvFilter
from ._utils import pad_2d, pad_3d, unpad_3d


def hv_loss(img: torch.Tensor, weighting: float = 0.5) -> torch.Tensor:
    """Sparse Hessian regularization term (2D)."""
    dxx2 = torch.square(-img[:, :, 2:, 1:-1] + 2 * img[:, :, 1:-1, 1:-1] - img[:, :, :-2, 1:-1])
    dyy2 = torch.square(-img[:, :, 1:-1, 2:] + 2 * img[:, :, 1:-1, 1:-1] - img[:, :, 1:-1, :-2])
    dxy2 = torch.square(img[:, :, 2:, 2:] - img[:, :, 2:, 1:-1] - img[:, :, 1:-1, 2:] +
                        img[:, :, 1:-1, 1:-1])
    h_v = torch.sqrt(weighting * weighting * (dxx2 + dyy2 + 2 * dxy2) +
                     (1 - weighting) * (1 - weighting) * torch.square(img[:, :, 1:-1, 1:-1]))
    return torch.mean(h_v)


def hv_loss_3d(img: torch.Tensor, delta: float = 1, weighting: float = 0.5) -> torch.Tensor:
    """Sparse Hessian regularization term (3D)."""
    img_ = img[:, :, 1:-1, 1:-1, 1:-1]
    d11 = -img[:, :, 1:-1, 1:-1, 2:] + 2*img_ - img[:, :, 1:-1, 1:-1, :-2]
    d22 = -img[:, :, 1:-1, 2:, 1:-1] + 2*img_ - img[:, :, 1:-1, :-2, 1:-1]
    d33 = delta*delta*(-img[:, :, 2:, 1:-1, 1:-1] + 2*img_ - img[:, :, :-2, 1:-1, 1:-1])
    d12_d21 = img[:, :, 1:-1, 2:, 2:] - img[:, :, 1:-1, 1:-1, 2:] - img[:, :, 1:-1, 2:, 1:-1] + img_
    d13_d31 = delta*(img[:, :, 2:, 1:-1, 2:] - img[:, :, 1:-1, 1:-1, 2:]
                     - img[:, :, 2:, 1:-1, 1:-1] + img_)
    d23_d32 = delta*(img[:, :, 2:, 2:, 1:-1] - img[:, :, 1:-1, 2:, 1:-1]
                     - img[:, :, 2:, 1:-1, 1:-1] + img_)

    h_v = torch.square(weighting*d11) + torch.square(weighting*d22) + torch.square(
        weighting*d33) + 2 * torch.square(weighting*d12_d21) + 2 * torch.square(
        weighting*d13_d31) + 2 * torch.square(weighting*d23_d32) + torch.square((1-weighting)*img_)

    return torch.mean(torch.sqrt(h_v))


def dataterm_deconv(blurry_image: torch.Tensor,
                    deblurred_image: torch.Tensor,
                    psf: torch.Tensor) -> torch.Tensor:
    """Deconvolution L2 data-term (2D)."""
    conv_op = torch.nn.Conv2d(1, 1, kernel_size=psf.shape[2],
                              stride=1,
                              padding=int((psf.shape[2] - 1) / 2),
                              bias=False)
    with torch.no_grad():
        conv_op.weight = torch.nn.Parameter(psf)
    mse = torch.nn.MSELoss()
    return mse(blurry_image, conv_op(deblurred_image))


class DataTermDeconv3D(torch.autograd.Function):
    """Deconvolution L2 data term (3D) with manual backward."""
    @staticmethod
    def forward(ctx, deblurred_image, blurry_image, fft_blurry_image, fft_psf, adjoint_otf):
        fft_deblurred_image = torch.fft.fftn(deblurred_image)
        ctx.save_for_backward(deblurred_image, fft_deblurred_image, fft_blurry_image, fft_psf,
                              adjoint_otf)
        conv_deblured_image = torch.real(torch.fft.ifftn(fft_deblurred_image * fft_psf))
        mse = torch.nn.MSELoss()
        return mse(blurry_image, conv_deblured_image)

    @staticmethod
    def backward(ctx, grad_output):
        deblurred_image, fft_deblurred_image, fft_blurry_image, \
        fft_psf, adjoint_otf = ctx.saved_tensors

        real_tmp = fft_psf.real * fft_deblurred_image.real - \
                   fft_psf.imag * fft_deblurred_image.imag - fft_blurry_image.real
        imag_tmp = fft_psf.real * fft_deblurred_image.imag + \
                   fft_psf.imag * fft_deblurred_image.real - fft_blurry_image.imag

        residue_image_real = adjoint_otf.real * real_tmp - adjoint_otf.imag * imag_tmp
        residue_image_imag = adjoint_otf.real * imag_tmp + adjoint_otf.imag * real_tmp

        grad_ = torch.real(torch.fft.ifftn(
                           torch.complex(residue_image_real,
                                         residue_image_imag))) / torch.numel(deblurred_image)
        return grad_output * grad_, None, None, None, None


class Spitfire(SDeconvFilter):
    """Variational deconvolution using the Spitfire algorithm.

    Parameters
    ----------
    psf : torch.Tensor
        Point spread function.
    weight : float
        Model weight between hessian and sparsity, in (0, 1).
    delta : float
        For 3D images: resolution delta between XY and Z.
    reg : float
        Regularization weight, in [0, 1].
    gradient_step : float
        Gradient descent step.
    precision : float
        Stop criterion.
    pad : int or tuple
        Image padding.
    """
    def __init__(self,
                 psf: torch.Tensor,
                 weight: float = 0.6,
                 delta: float = 1,
                 reg: float = 0.995,
                 gradient_step: float = 0.01,
                 precision: float = 1e-7,
                 pad: int | tuple[int, int] | tuple[int, int, int] = 0):
        super().__init__()
        self.psf = psf.to(SSettings.instance().device)
        self.weight = weight
        self.reg = reg
        self.precision = precision
        self.delta = delta
        self.pad = pad
        self.niter_ = 0
        self.max_iter_ = 2500
        self.gradient_step_ = gradient_step
        self.loss_ = None

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 2:
            return self.run_2d(image)
        if image.ndim == 3:
            return self.run_3d(image)
        raise ValueError("Spitfire can process only 2D or 3D tensors")

    def run_2d(self, image: torch.Tensor) -> torch.Tensor:
        self.progress(0)
        mini = torch.min(image) + 1e-5
        maxi = torch.max(image)
        image = (image-mini)/(maxi-mini)

        image_pad, psf_pad, padding = pad_2d(image, self.psf/torch.sum(self.psf), self.pad)
        psf_pad = self.psf/torch.sum(self.psf)

        image_pad = image_pad.view(1, 1, image_pad.shape[0], image_pad.shape[1])
        psf_pad = psf_pad.view(1, 1, psf_pad.shape[0], psf_pad.shape[1])
        deconv_image = image_pad.detach().clone()
        deconv_image.requires_grad = True
        optimizer = torch.optim.Adam([deconv_image], lr=self.gradient_step_)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        previous_loss = 9e12
        count_eq = 0
        self.niter_ = 0
        loss = 0

        for i in range(self.max_iter_):
            self.progress(int(100*i/self.max_iter_))
            self.niter_ += 1
            optimizer.zero_grad()
            loss = self.reg * dataterm_deconv(image_pad, deconv_image, psf_pad) + \
                (1-self.reg) * hv_loss(deconv_image, self.weight)
            if abs(loss - previous_loss) < self.precision:
                count_eq += 1
            else:
                previous_loss = loss
                count_eq = 0
            if count_eq > 5:
                break
            loss.backward()
            optimizer.step()
            scheduler.step()
        self.loss_ = loss
        self.progress(100)
        deconv_image = (maxi-mini)*deconv_image + mini
        deconv_image = deconv_image.view(deconv_image.shape[2], deconv_image.shape[3])
        if image_pad.shape[2] != image.shape[0] and image_pad.shape[3] != image.shape[1]:
            return deconv_image[padding[0]: -padding[0], padding[1]: -padding[1]]
        return deconv_image

    @staticmethod
    def otf_3d(psf: torch.Tensor) -> torch.Tensor:
        psf_roll = torch.roll(psf, int(-psf.shape[0] / 2), dims=0)
        psf_roll = torch.roll(psf_roll, int(-psf.shape[1] / 2), dims=1)
        psf_roll = torch.roll(psf_roll, int(-psf.shape[2] / 2), dims=2)
        psf_roll.view(1, psf.shape[0], psf.shape[1], psf.shape[2])
        return torch.fft.fftn(psf_roll)

    @staticmethod
    def adjoint_otf(psf: torch.Tensor) -> torch.Tensor:
        adjoint_psf = torch.flip(psf, [0, 1, 2])
        adjoint_psf = torch.roll(adjoint_psf, -int(psf.shape[0] - 1) % 2, dims=0)
        adjoint_psf = torch.roll(adjoint_psf, -int(psf.shape[1] - 1) % 2, dims=1)
        adjoint_psf = torch.roll(adjoint_psf, -int(psf.shape[2] - 1) % 2, dims=2)
        adjoint_psf = torch.roll(adjoint_psf, int(-psf.shape[0] / 2), dims=0)
        adjoint_psf = torch.roll(adjoint_psf, int(-psf.shape[1] / 2), dims=1)
        adjoint_psf = torch.roll(adjoint_psf, int(-psf.shape[2] / 2), dims=2)
        return torch.fft.fftn(adjoint_psf)

    def run_3d(self, image: torch.Tensor) -> torch.Tensor:
        self.progress(0)
        mini = torch.min(image) + 1e-5
        maxi = torch.max(image)
        image = (image-mini)/(maxi-mini)
        image_pad, psf_pad, padding = pad_3d(image, self.psf / torch.sum(self.psf), self.pad)

        deconv_image = image_pad.detach().clone()
        image_pad = image_pad.view(1, 1, image_pad.shape[0], image_pad.shape[1], image_pad.shape[2])
        deconv_image = deconv_image.view(1, 1, deconv_image.shape[0],
                                         deconv_image.shape[1], deconv_image.shape[2])
        deconv_image.requires_grad = True
        optimizer = torch.optim.Adam([deconv_image], lr=self.gradient_step_)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        previous_loss = 9e12
        count_eq = 0
        self.niter_ = 0

        fft_psf = Spitfire.otf_3d(psf_pad)
        adjoint_otf = Spitfire.adjoint_otf(psf_pad)
        fft_image = torch.fft.fftn(image_pad)
        dataterm_ = DataTermDeconv3D.apply
        loss = 0
        for i in range(self.max_iter_):
            self.progress(int(100*i/self.max_iter_))
            self.niter_ += 1
            optimizer.zero_grad()
            loss = self.reg * dataterm_(deconv_image, image_pad,
                                        fft_image, fft_psf, adjoint_otf) + (
                        1 - self.reg) * hv_loss_3d(deconv_image, self.delta, self.weight)
            if loss > previous_loss:
                break
            if abs(loss - previous_loss) < self.precision:
                count_eq += 1
            else:
                previous_loss = loss
                count_eq = 0
            if count_eq > 5:
                break
            loss.backward()
            optimizer.step()
            scheduler.step()
        self.loss_ = loss
        self.progress(100)
        deconv_image = deconv_image.view(image_pad.shape[2],
                                         image_pad.shape[3],
                                         image_pad.shape[4])
        deconv_image = (maxi-mini)*deconv_image + mini
        if image_pad.shape[2] != image.shape[0] and image_pad.shape[3] != image.shape[1] and \
                image_pad.shape[4] != image.shape[2]:
            return unpad_3d(deconv_image, padding)
        return deconv_image
