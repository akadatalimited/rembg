import os
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from PIL.Image import Image as PILImage

import cupy as cp  # For GPU operations

import asyncio  # Import asyncio for parallel processing
from concurrent.futures import ThreadPoolExecutor

class BaseSession:
    """This is a base class for managing a session with a machine learning model.
    Forked by AKADATA LIMITED and enhanced by Andrew Smalley.
    """

    def __init__(
        self,
        model_name: str,
        sess_opts: ort.SessionOptions,
        providers=None,
        *args,
        **kwargs
    ):
        """Initialize an instance of the BaseSession class."""
        self.model_name = model_name

        self.providers = []

        # Add CUDA provider first for GPU acceleration
        if ort.get_available_providers():
            if "CUDAExecutionProvider" in ort.get_available_providers():
                self.providers.append("CUDAExecutionProvider")
            else:
                self.providers.extend(ort.get_available_providers())

        # Create Inference Session with prioritized CUDA provider
        self.inner_session = ort.InferenceSession(
            str(self.__class__.download_models(*args, **kwargs)),
            providers=self.providers,
            sess_options=sess_opts,
        )

    def normalize(
        self,
        img: PILImage,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        size: Tuple[int, int],
        *args,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        # Convert the image and move to GPU
        im = img.convert("RGB").resize(size, Image.Resampling.LANCZOS)

        # Move normalization to GPU using CuPy
        im_ary = cp.asarray(im) / 255.0

        tmpImg = cp.zeros((im_ary.shape[0], im_ary.shape[1], 3), dtype=cp.float32)
        tmpImg[:, :, 0] = (im_ary[:, :, 0] - mean[0]) / std[0]
        tmpImg[:, :, 1] = (im_ary[:, :, 1] - mean[1]) / std[1]
        tmpImg[:, :, 2] = (im_ary[:, :, 2] - mean[2]) / std[2]

        tmpImg = tmpImg.transpose((2, 0, 1))

        return {
            self.inner_session.get_inputs()[0]
            .name: cp.expand_dims(tmpImg, 0)
            .astype(cp.float32)
            .get()  # Convert back to numpy to be used by ONNX Runtime
        }

    def predict_batch(self, imgs: List[PILImage], *args, **kwargs) -> List[List[PILImage]]:
        """
        Predict the output masks for a batch of input images.

        Parameters:
            imgs (List[PILImage]): A list of input images.

        Returns:
            List[List[PILImage]]: A list of lists containing output masks for each input image.
        """
        inputs = [self.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (1024, 1024)) for img in imgs]
        ort_outs = [self.inner_session.run(None, input) for input in inputs]
        
        results = []
        for img, ort_out in zip(imgs, ort_outs):
            pred = self.sigmoid(ort_out[0][:, 0, :, :])
            ma = cp.max(pred)
            mi = cp.min(pred)
            pred = (pred - mi) / (ma - mi)
            pred = cp.squeeze(pred)
            mask = Image.fromarray((pred.get() * 255).astype("uint8"), mode="L")
            results.append([mask.resize(img.size, Image.Resampling.LANCZOS)])

        return results

    def sigmoid(self, mat):
        """
        Apply the sigmoid function to the given matrix.
        
        Parameters:
            mat (cp.ndarray): The input matrix.
        
        Returns:
            cp.ndarray: The result after applying the sigmoid function.
        """
        return 1 / (1 + cp.exp(-mat))

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        raise NotImplementedError

    async def predict_async(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, self.predict, img, *args, **kwargs)
        return result

    @classmethod
    def checksum_disabled(cls, *args, **kwargs):
        return os.getenv("MODEL_CHECKSUM_DISABLED", None) is not None

    @classmethod
    def u2net_home(cls, *args, **kwargs):
        return os.path.expanduser(
            os.getenv(
                "U2NET_HOME", os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".u2net")
            )
        )

    @classmethod
    def download_models(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def name(cls, *args, **kwargs):
        raise NotImplementedError
