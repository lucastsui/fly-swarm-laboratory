import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from .cluster_protocol import ParameterModel, optimizer_for, pack, unpack, signature
from .cluster_server import Coordinator, serve
from .cluster_protocol import ClusterClient


def weights():
    return {"encoder": nn.Linear(22, 8).state_dict(), "log_gain": torch.zeros(20), "bias": torch.zeros(20),
            "readout": nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 64), nn.Tanh()).state_dict(),
            "actor": nn.Linear(64, 6).state_dict(), "critic": nn.Linear(64, 1).state_dict()}


class ClusterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.bootstrap = {"weights": weights(), "updates": 7}
        self.config = {w: {"label": w, "host": "test", "environmentCount": 2, "flyCount": 4} for w in ("laptop", "spark1", "spark2")}
        self.server = Coordinator(self.bootstrap, {}, Path(self.temp.name), self.config)
        self.server.control({"action": "start"})

    def tearDown(self):
        self.temp.cleanup()

    def message(self, worker="laptop", request="one"):
        s = self.server
        return {"worker": worker, "id": request, "signature": s.layout, "generation": s.generation,
                "version": s.version, "gradient": torch.ones_like(s.flat) * .01, "samples": 32}

    def test_shared_optimizer_matches_direct_step_and_deduplicates(self):
        direct = ParameterModel(self.bootstrap["weights"])
        optimizer = optimizer_for(direct)
        message = self.message()
        for p in direct.parameters():
            p.grad = torch.ones_like(p) * .01
        torch.nn.utils.clip_grad_norm_(direct.parameters(), 1)
        optimizer.step()
        result = self.server.update(message)
        self.assertTrue(result["accepted"])
        self.assertTrue(torch.equal(result["parameters"], torch.nn.utils.parameters_to_vector(direct.parameters())))
        self.server.update(message)
        self.assertEqual(self.server.version, 8)
        self.assertEqual(self.server.accepted["laptop"], 1)
        self.server.update(self.message("spark1", "two"))
        self.server.update(self.message("spark2", "three"))
        self.assertEqual(self.server.version, 10)
        self.assertEqual(sum(self.server.samples.values()), 96)

    def test_stale_invalid_paused_and_wrong_generation(self):
        message = self.message()
        message["version"] -= 17
        self.assertFalse(self.server.update(message)["accepted"])
        message = self.message(request="nan")
        message["gradient"][0] = float("nan")
        with self.assertRaises(ValueError):
            self.server.update(message)
        self.server.control({"action": "pause"})
        self.assertFalse(self.server.update(self.message(request="paused"))["accepted"])
        self.server.control({"action": "start"})
        message = self.message(request="generation")
        message["generation"] = "wrong"
        self.assertFalse(self.server.update(message)["accepted"])
        self.assertEqual(self.server.version, 7)

    def test_atomic_restore_and_reset_retains_model(self):
        self.server.update(self.message())
        self.server.save()
        restored = Coordinator(self.bootstrap, {}, Path(self.temp.name), self.config)
        self.assertTrue(torch.equal(restored.flat, self.server.flat))
        self.assertEqual(restored.version, 8)
        self.assertFalse(restored.running)
        restored.watch["spark2"] = {"environment": 1, "fly": 3}
        old = restored.generation
        restored.control({"action": "reset", "worker": "spark2", "environmentCount": 1, "flyCount": 1})
        self.assertNotEqual(old, restored.generation)
        self.assertEqual(restored.watch["spark2"], {"environment": 0, "fly": 0})
        self.assertTrue(torch.equal(restored.flat, self.server.flat))
        self.assertTrue((Path(self.temp.name) / "archives" / old / "cluster-state.pt").exists())

    def test_authenticated_transport(self):
        http = serve(self.server, "test-token", "127.0.0.1", 0)
        url = f"http://127.0.0.1:{http.server_port}"
        try:
            client = ClusterClient(url, "test-token")
            initial = client.tensor("/worker/parameters?worker=spark1")
            self.assertEqual(initial["signature"], signature(self.server.model))
            updated = client.tensor("/worker/update", self.message("spark1"))
            self.assertTrue(updated["accepted"])
            self.assertEqual(unpack(pack(updated))["weightHash"], self.server.weight_hash)
            with self.assertRaises(Exception):
                ClusterClient(url, "bad").json("/api/engine/health")
        finally:
            http.shutdown()
            http.server_close()


if __name__ == "__main__":
    unittest.main()
