"""Fast public checks; no pretrained weights or external images required."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eciqa.components import CandidateCrossAttention
from eciqa.contract import ARCHITECTURE, validate_config, validate_payload
from eciqa.inference import make_pair_sample
from eciqa.metrics import compute_metrics
from eciqa.train import quality_objective


class ReleaseContractTests(unittest.TestCase):
    def test_reject_wrong_architecture(self):
        for change in ({'use_global_cls': True}, {'cross_view_attention': False},
                       {'train_tiles': 16}, {'exposure_prompt': True}, {'prompt_only': False}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config({**ARCHITECTURE, **change})

    def test_reject_unversioned_checkpoint(self):
        with self.assertRaises(ValueError):
            validate_payload({'model': {}})

    def test_soft_labels_equal_two_class_cross_entropy(self):
        logits = torch.tensor([[1.3, -0.6]], requires_grad=True)
        target = torch.tensor(0.73)
        actual = quality_objective({'quality_logit': logits[0, 0]-logits[0, 1],
                                    'prediction': logits.softmax(-1)[0, 0]}, target)
        distribution = torch.stack((target, 1-target))[None]
        expected = -(distribution * torch.log_softmax(logits, dim=-1)).sum()
        self.assertTrue(torch.allclose(actual, expected))
        self.assertTrue(torch.allclose(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                                       torch.autograd.grad(expected, logits)[0]))

    def test_original_resolution_boundaries_and_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            array = np.arange(225*447*3, dtype=np.uint32).reshape(225,447,3).astype(np.uint8)
            Image.fromarray(array).save(root/'input.png')
            Image.fromarray(array).save(root/'candidate.png')
            sample = make_pair_sample(root/'input.png',root/'candidate.png',dict(ARCHITECTURE))
            self.assertEqual(len(sample['source']), 5)  # thumbnail + ceil(225/224)*ceil(447/224)
            self.assertTrue(torch.equal(sample['source'],sample['candidate']))
            self.assertTrue(sample['full_native_coverage'])
            self.assertAlmostEqual(float(sample['valid_fraction'][1:].sum())*256,225*447,places=3)
            self.assertTrue(torch.equal(sample['masks'],sample['valid_fraction']>0))

    def test_masked_source_tokens_do_not_affect_cross_attention(self):
        torch.manual_seed(7)
        attention = CandidateCrossAttention(8,heads=2).eval()
        query,source = torch.randn(1,3,8),torch.randn(1,4,8)
        mask = torch.tensor([[True,True,False,False]])
        perturbed = source.clone()
        perturbed[:,2:] = 1000
        self.assertTrue(torch.allclose(attention(query,source,input_mask=mask),
                                       attention(query,perturbed,input_mask=mask)))

    def test_constant_correlation_not_fabricated(self):
        metrics = compute_metrics([0.1,0.4,0.8],[0.5,0.5,0.5])
        self.assertIsNone(metrics['srcc'])
        self.assertIsNone(metrics['plcc'])
        json.dumps(metrics,allow_nan=False)


if __name__ == '__main__':
    unittest.main()
