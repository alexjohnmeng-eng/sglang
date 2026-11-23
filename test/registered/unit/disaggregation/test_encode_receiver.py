"""Unit tests for the encode-disaggregation receiver."""

import time
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.encoder.receiver import (
    MMReceiverBase,
    MMReceiverHTTP,
    WaitingListSync,
    WaitingMMRequestStatus,
    _select_mm_processor_prompt,
)
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestSelectMMProcessorPrompt(CustomTestCase):
    def test_tokens_in_prompt_is_list(self):
        """A tokens-in prompt reaches the processor as list[int], not array("q");
        HTTP sends input_text=None and the gRPC servicer sends ""."""
        processor = SimpleNamespace(prefer_tokenized_input=False)
        for input_text in (None, ""):
            with self.subTest(input_text=input_text):
                recv_req = SimpleNamespace(
                    input_text=input_text, input_ids=array("q", [1, 2, 3])
                )

                prompt = _select_mm_processor_prompt(recv_req, processor)

                self.assertIsInstance(prompt, list)
                self.assertEqual(prompt, [1, 2, 3])


class TestWaitingRequestTimeout(CustomTestCase):
    def test_a_result_reached_in_the_timeout_tick_is_kept(self):
        receiver = MMReceiverHTTP.__new__(MMReceiverHTTP)
        receiver.wait_timeout = 0.0
        receiver.waiting_by_rid = {}
        receiver.waiting_list_sync = WaitingListSync()
        receiver._drain_scheduler_embeddings = Mock()
        receiver.create_req = Mock()
        waiting_req = SimpleNamespace(
            rid="r",
            recv_req=SimpleNamespace(rid="r"),
            start_time=0.0,
            status=WaitingMMRequestStatus.PENDING,
            release_resources=Mock(),
            close_recv_socket=Mock(),
        )

        def recv():
            waiting_req.status = WaitingMMRequestStatus.SUCCESS

        waiting_req._try_recv_mm_data = recv
        receiver.waiting_list = [waiting_req]

        with patch("torch.distributed.all_reduce"):
            new_recv_reqs, abort_reqs = receiver._process_waiting_requests([], None)

        self.assertEqual(new_recv_reqs, [waiting_req.recv_req])
        self.assertEqual(abort_reqs, [])
        waiting_req.release_resources.assert_not_called()


class _RecordingGroup:
    def __init__(self, name, world_size, peer_values=None):
        self.name = name
        self.world_size = world_size
        self.cpu_group = name
        self.peer_values = peer_values

    def all_gather_object(self, values):
        return [values] + [self.peer_values] * (self.world_size - 1)


class _CompletedRequest:
    def __init__(self, rid):
        self.rid = rid
        self.recv_req = SimpleNamespace(rid=rid, encoder_req_id=None)
        self.status = WaitingMMRequestStatus.SUCCESS
        self.start_time = time.time()

    def _try_recv_mm_data(self):
        pass

    def close_recv_socket(self):
        pass


def _scheduler_rank(waiting_list, waiting_list_sync):
    return SimpleNamespace(
        waiting_list=waiting_list,
        waiting_by_rid={request.rid: request for request in waiting_list},
        wait_timeout=float("inf"),
        waiting_list_sync=waiting_list_sync,
        _drain_scheduler_embeddings=lambda: None,
        _sync_fail_info_across_tp=lambda request: None,
        create_req=lambda recv_req: recv_req,
    )


class TestDpAttentionWaitingList(CustomTestCase):
    def test_encoder_waits_for_registrations_from_this_request_shard_only(self):
        request = Mock(spec=TokenizedGenerateReqInput)
        request.rid = "a"
        request.need_wait_for_mm_inputs = True
        request.encoder_urls = ["http://encoder"]
        completed = _CompletedRequest("a")
        completed.send_encode_request = Mock()
        waiting_cls = Mock(return_value=completed)
        rank = _scheduler_rank([], WaitingListSync())
        rank.encode_urls = request.encoder_urls
        rank.mm_processor = Mock()
        rank.model_type = "deepseek_v41"
        rank.hostname = "localhost"
        rank.scheduler_recv_socket = Mock()
        rank.scheduler_embedding_port = 12345
        ready, aborted = MMReceiverBase._process_waiting_requests(
            rank, [request], waiting_cls
        )
        self.assertEqual(waiting_cls.call_args.kwargs["receive_count"], 1)
        self.assertEqual([r.rid for r in ready], ["a"])
        self.assertEqual(aborted, [])

    def test_dp_ranks_with_their_own_requests_sync_only_their_shard(self):
        """With DP attention each DP rank of a TP group holds different
        requests; a status all-reduce over the whole TP group got tensors of
        different lengths from its ranks and aborted gloo, and the encoder
        waited for registrations from every TP rank."""
        calls = []

        def record(tensor, op=None, group=None):
            calls.append((group, tensor.numel()))

        # TP4 x DP4 with attn_tp = attn_cp = 1: every waiting list is one rank's.
        parallel = SimpleNamespace(
            attn_dp_enabled=True,
            attn_tp_group=_RecordingGroup("attn_tp", 1),
            attn_cp_group=_RecordingGroup("attn_cp", 1),
            tp_group=_RecordingGroup("full_tp", 4),
        )
        busy_sync = WaitingListSync.for_scheduler(parallel)
        self.assertEqual(busy_sync.size, 1)
        busy = _scheduler_rank([_CompletedRequest("a")], busy_sync)
        idle = _scheduler_rank([], WaitingListSync.for_scheduler(parallel))
        with patch("torch.distributed.all_reduce", record):
            ready_busy, _ = MMReceiverBase._process_waiting_requests(
                busy, [], waiting_cls=None
            )
            ready_idle, _ = MMReceiverBase._process_waiting_requests(
                idle, [], waiting_cls=None
            )

        self.assertEqual(calls, [])
        self.assertEqual([req.rid for req in ready_busy], ["a"])
        self.assertEqual(ready_idle, [])
        # Without DP attention the whole TP group shares one waiting list.
        parallel.attn_dp_enabled = False
        self.assertEqual(WaitingListSync.for_scheduler(parallel).size, 4)

    def test_shard_collectives_run_along_attn_tp_then_attn_cp(self):
        calls = []

        def record(tensor, op=None, group=None):
            calls.append(group)

        shard = WaitingListSync(
            (
                _RecordingGroup("attn_tp", 2, peer_values=["tp-peer"]),
                _RecordingGroup("attn_cp", 2, peer_values=["cp-a", "cp-b"]),
            )
        )
        rank = _scheduler_rank([_CompletedRequest("a")], shard)
        with patch("torch.distributed.all_reduce", record):
            MMReceiverBase._process_waiting_requests(rank, [], waiting_cls=None)

        self.assertEqual(calls, ["attn_tp", "attn_cp"])
        self.assertEqual(shard.size, 4)
        self.assertEqual(
            shard.all_gather_object("me"), ["me", "tp-peer", "cp-a", "cp-b"]
        )


if __name__ == "__main__":
    unittest.main()
