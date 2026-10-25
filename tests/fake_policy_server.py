"""Test helper: a stand-in for the VPP policy server (AtomicVPP_LIBERO step3_deploy_real_ur5.py), same wire protocol
(policy.send_msg / recv_msg), no model. It describes the training data of deploy.yml's settings, and answers each
request with a chunk that moves STEP_M along +x per step from the state, keeps the state's rotation, and closes the
gripper from step CLOSE_FROM on. Every request is kept (requests; with --log also one JSON line each in that file).

  python -m tests.fake_policy_server --port 47999 [--log requests.jsonl]
"""
import argparse
import json
import socket
import threading

import numpy as np

import policy
import vpp

STEP_M = 0.005  # 5 cm/s at 10 Hz, below MAX_LINEAR_M_S
CLOSE_FROM = 5
MODEL = {"checkpoint": "fake", "weights": "fake", "epoch": 0, "train_steps": 0, "val": None, "video_model": "fake",
         "action_mode": "relative", "action_seq_len": 10, "obs_seq_len": 1, "cameras": ["fixed", "wrist"],
         "crop": {"fixed": [150, 100, 500, 450]}, "size": 256, "fps": 10, "state_layout": vpp.STATE_LAYOUT,
         "instructions": ["t"], "episodes": 0, "device": "fake"}


class FakeServer:
    def __init__(self, port=0, model=None, log=None):
        self.model, self.log, self.requests = model or MODEL, log, []
        self.srv = socket.create_server(("127.0.0.1", port))
        self.port = self.srv.getsockname()[1]
        threading.Thread(target=self.serve, daemon=True).start()

    def serve(self):
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:  # closed
                return
            threading.Thread(target=self.handle, args=(conn,), daemon=True).start()

    def handle(self, conn):
        try:
            while True:
                header, blobs = policy.recv_msg(conn)
                if header["type"] == "hello":
                    policy.send_msg(conn, {"type": "hello", "protocol": policy.PROTOCOL, "model": self.model})
                    continue
                images = [np.frombuffer(b, np.uint8).reshape(i["shape"]) for b, i in zip(blobs, header["images"])]
                state = np.asarray(header["state"])
                n = self.model["action_seq_len"]
                actions = np.tile(state, (n, 1))
                actions[:, 0] += STEP_M * np.arange(n)
                actions[:, 9] = (np.arange(n) >= CLOSE_FROM).astype(float)
                record = {**header, "image_shapes": [list(i.shape) for i in images],
                          "image_means": [[float(v) for v in i.reshape(-1, 3).mean(0)] for i in images]}
                self.requests.append(record)
                if self.log:
                    with open(self.log, "a") as f:
                        f.write(json.dumps(record) + "\n")
                policy.send_msg(conn, {"type": "actions", "id": header["id"], "actions": actions.tolist(),
                                       "relative": (actions - np.r_[state[:9], 0]).tolist(), "infer_ms": 1.0})
        except (ConnectionError, OSError):
            conn.close()

    def close(self):
        self.srv.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--log")
    args = p.parse_args()
    server = FakeServer(args.port, log=args.log)
    print(f"fake policy server on 127.0.0.1:{server.port}", flush=True)
    threading.Event().wait()
