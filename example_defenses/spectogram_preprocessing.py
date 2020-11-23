import numpy as np
from scipy import signal
from typing import Optional, Tuple

from art.defences.preprocessor import Preprocessor


class Spectogram(Preprocessor):
    def __init__(self) -> None:
        pass

    def __call__(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        samplerate_hz = 16000  # sample rate for wav files
        window_time_secs = 0.030  # 30 ms windows for frequency transforms
        num_samples_overlap = int(0.025 * samplerate_hz)  # 25 ms of overlap

        # -- NORMALIZATION PARAMETERS --#
        zscore_mean = -0.7731548539849517
        zscore_std = 3.5610712683198624
        scale_max = 15.441861
        scale_min = -4.6051702

        # Construct window
        window_num_samples = int(window_time_secs * samplerate_hz)
        window = signal.get_window(("tukey", 0.25), window_num_samples)

        def normalize_spectrogram(s):
            """ Normalize spectrogram s:
            1. s_ = np.log(s + 0.01)
            2. s_ = zscores(s_)
            3. s_ = minmax_scale(s_)

            Return normalized spectrogram s_ in range [-1, 1] with mean ~ 0 and std ~ 1
            """
            s_ = np.log(s + 0.01)
            s_ = (s_ - zscore_mean) / zscore_std
            s_ = (s_ - scale_min) / (scale_max - scale_min)
            return s_

        def spectrogram_241(samples):
            """ Return vector of frequences (f), vector of times (t), and 2d matrix spectrogram (s)
            for input audio samples.
            """
            # Construct spectrogram (f = frequencies array, t = times array, s = 2d spectrogram [f x t])
            f, t, s = signal.spectrogram(
                samples, samplerate_hz, window=window, noverlap=num_samples_overlap
            )

            # Normalize spectrogram
            s = normalize_spectrogram(s)

            return f, t, s

        def segment(x, y, n_time_bins):
            """
            Return segmented batch of spectrograms and labels

            x is of shape (N,241,T), representing N spectrograms, each with 241 frequency bins
            and T time bins that's variable, depending on the duration of the corresponding
            raw audio.

            The model accepts a fixed size spectrogram, so data needs to be segmented for a
            fixed number of time_bins.
            """
            x_seg = []
            for xt in x:
                n_seg = int(xt.shape[1] / n_time_bins)
                xt = xt[:, : n_seg * n_time_bins]
                for ii in range(n_seg):
                    x_seg.append(xt[:, ii * n_time_bins : (ii + 1) * n_time_bins])
            x_seg = np.array(x_seg)
            x_seg = np.expand_dims(x_seg, -1)
            return x_seg, y

        quantization = 2 ** 15
        n_tbins = 100  # number of time bins in spectrogram input to model
        if x.dtype == np.float32:
            x = [x]

        outputs = []
        for aud in x:
            aud = np.squeeze(
                (aud * quantization).astype(np.int64)
            )  # Reverse canonical preprocessing
            _, _, s = spectrogram_241(aud)
            outputs.append(s)
        return segment(outputs, y, n_tbins)

    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return grad

    def apply_fit(self):
        return True

    def apply_predict(self):
        return True

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        No parameters to learn for this method; do nothing.
        """
        pass
