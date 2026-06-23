import os
import unittest

import numpy as np
import torch

import vae_hooks


class FakeLatentDist:
    def __init__(self, x):
        self.x = x

    def mode(self):
        return torch.ones(
            (self.x.shape[0], 4, max(1, self.x.shape[2] // 8), max(1, self.x.shape[3] // 8)),
            device=self.x.device,
            dtype=self.x.dtype,
        )


class FakeEncodeOutput:
    def __init__(self, x):
        self.latent_dist = FakeLatentDist(x)


class FakeConfig:
    scaling_factor = 0.5


class FakeVAE:
    load_calls = []
    encoded_inputs = []

    config = FakeConfig()

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        cls.load_calls.append((model_name, kwargs))
        return cls()

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def parameters(self):
        return []

    def encode(self, x):
        self.encoded_inputs.append(x)
        return FakeEncodeOutput(x)


class VaeHooksTest(unittest.TestCase):
    def setUp(self):
        self.old_env = dict(os.environ)
        FakeVAE.load_calls = []
        FakeVAE.encoded_inputs = []
        vae_hooks._VAE_CACHE.clear()
        vae_hooks.AutoencoderKL = FakeVAE

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.old_env)
        vae_hooks._VAE_CACHE.clear()

    def test_encode_sd_vae_uses_env_config_preprocesses_and_caches(self):
        os.environ["FREQ_ANCHOR_DEVICE"] = "cpu"
        os.environ["FREQ_ANCHOR_DTYPE"] = "fp32"
        os.environ["FREQ_ANCHOR_VAE_MODEL"] = "fake/model"
        os.environ["FREQ_ANCHOR_VAE_SUBFOLDER"] = "vae"

        img = np.zeros((16, 24, 3), dtype=np.float32)
        img[..., 0] = 0.25
        img[..., 1] = 0.5
        img[..., 2] = 1.0

        z_a = vae_hooks.encode_sd_vae(img)
        z_b = vae_hooks.encode_sd_vae(img)

        self.assertEqual(len(FakeVAE.load_calls), 1)
        model_name, kwargs = FakeVAE.load_calls[0]
        self.assertEqual(model_name, "fake/model")
        self.assertEqual(kwargs["subfolder"], "vae")
        self.assertEqual(kwargs["torch_dtype"], torch.float32)
        self.assertEqual(z_a.dtype, torch.float32)
        self.assertTrue(torch.allclose(z_a, torch.full_like(z_a, 0.5)))
        self.assertTrue(torch.allclose(z_b, z_a))

        encoded = FakeVAE.encoded_inputs[0]
        self.assertEqual(tuple(encoded.shape), (1, 3, 16, 24))
        self.assertEqual(encoded.dtype, torch.float32)
        self.assertAlmostEqual(float(encoded[0, 0, 0, 0]), -0.5)
        self.assertAlmostEqual(float(encoded[0, 1, 0, 0]), 0.0)
        self.assertAlmostEqual(float(encoded[0, 2, 0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
