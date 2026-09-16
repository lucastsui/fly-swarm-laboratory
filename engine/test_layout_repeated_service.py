import unittest
from .layout_repeated_service import aggregate


class RepeatedServiceTests(unittest.TestCase):
    def test_unique_layouts_not_replicates(self):
        a={'trials':[{'kind':'wide','seed':1,'products':4,'sustainedSuccess':True}]}
        b={'trials':[{'kind':'wide','seed':1,'products':0,'sustainedSuccess':False}]}
        r=aggregate([a,b])['wide']
        self.assertEqual(r['uniqueLayouts'],1)
        self.assertEqual(r['repeats'],2)
        self.assertEqual(r['layoutsSustainedInEveryRepeat'],0)
        self.assertEqual(r['layoutsWithInconsistentSustainedOutcome'],1)

    def test_mismatched_worlds(self):
        with self.assertRaises(ValueError):
            aggregate([{'trials':[{'kind':'wide','seed':1}]},{'trials':[]}])


if __name__=='__main__':
    unittest.main()
