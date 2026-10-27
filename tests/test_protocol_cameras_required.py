"""Both cameras are required on the policy wire. Every modality config in configs/gr00t lists ego_view and wrist, every
sim env renders both, and decode_request (mock server and GR00T shim) answers a request without either with a 400.
encode_request used to send an ego-only request anyway, so the gap only showed up as that 400 from the server; it now
refuses the observation with a ProtocolError that names the missing camera, before anything is sent."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from arrival.policy_server import gr00t_shim
from arrival.policy_server.mock_server import MockPolicyServer
from arrival.policy_server.protocol import VIDEO_KEYS, ProtocolError, decode_request, encode_request
from arrival.sim.policy import VLAClient

STATE = np.array([0.1, -0.2, 0.3, -1.5, 0.0, 1.2, 0.4, 0.04])


def _obs(*cameras):
    return {"state": STATE, **{c: np.full((8, 8, 3), 7, np.uint8) for c in cameras}}


@pytest.mark.parametrize("missing", VIDEO_KEYS)
def test_encoder_refuses_an_obs_without_either_camera(missing):
    with pytest.raises(ProtocolError, match=f"obs has no {missing} frame"):
        encode_request(_obs(*(c for c in VIDEO_KEYS if c != missing)), "open the drawer")


def test_client_refuses_an_obs_without_wrist_before_sending_it():
    with MockPolicyServer() as srv:
        c = VLAClient(srv.url)
        c.reset(None, {"instruction": "x"})
        with pytest.raises(ProtocolError, match="no wrist frame; the policy server needs ego_view and wrist"):
            c.act(_obs("ego_view"))  # used to go out and come back as HTTP 400 "missing video.wrist"


@pytest.mark.parametrize("missing", VIDEO_KEYS)
def test_decoder_still_requires_both_cameras(missing):
    body = encode_request(_obs(*VIDEO_KEYS), "open the drawer")
    del body[f"video.{missing}"]
    with pytest.raises(ProtocolError, match=f"missing video.{missing}"):
        decode_request(body)


def test_shim_answers_400_to_an_ego_only_request_and_never_calls_the_policy():
    seen = []

    class Fake:  # stands in for Gr00tPolicy
        def get_action(self, obs):
            seen.append(sorted(obs["video"]))
            return {"single_arm": np.zeros((1, 4, 7)), "gripper": np.zeros((1, 4, 1))}, {}

    def post(body):
        req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/act", data=json.dumps(body).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=5)

    full = encode_request(_obs(*VIDEO_KEYS), "open the drawer")
    ego_only = {k: v for k, v in full.items() if k != "video.wrist"}
    srv = ThreadingHTTPServer(("127.0.0.1", 0), gr00t_shim.make_handler(Fake()))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            post(ego_only)
        assert e.value.code == 400 and json.load(e.value) == {"error": "missing video.wrist"}
        assert seen == []
        assert post(full).status == 200
    finally:
        srv.shutdown()
        srv.server_close()
    assert seen == [["ego_view", "wrist"]]
