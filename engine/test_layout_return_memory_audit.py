import unittest
from .layout_return_memory_audit import cue_row


class MemoryWindowTests(unittest.TestCase):
    def test_exact_prefix_boundary_and_age(self):
        spec = dict(start=100,focusTick=180)
        self.assertFalse(cue_row(spec,100)['outsideDifferentiatedHistory'])
        self.assertTrue(cue_row(spec,99)['outsideDifferentiatedHistory'])
        self.assertEqual(cue_row(spec,180)['cueAgeSeconds'],0)
        self.assertEqual(cue_row(spec,100)['cueAgeSeconds'],4.)

    def test_missing_future_or_negative_cue_rejected(self):
        for cue in (None,-1,181):
            with self.assertRaises(ValueError): cue_row(dict(start=100,focusTick=180),cue)


if __name__ == '__main__': unittest.main()
