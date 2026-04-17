import datetime
import logging
import math
import random

import torch
import torch.distributed as dist
from nvidia_resiliency_ext.shared_utils.inject_fault import (
    Fault,
    clear_workload_exception,
    dispatch_fault_injection,
    maybe_raise_workload_exception,
)

logger = logging.getLogger(__name__)

rng = None


def get_fault_ranks(cli_args):
    global rng

    force_ranks = cli_args.fault_injector_ranks
    world_size = dist.get_world_size()

    if force_ranks is not None:
        assert cli_args.fault_injector_num_ranks is None, "Cannot specify both force_ranks and num_ranks"
        if ',' in force_ranks:
            fault_ranks = [int(r) for r in force_ranks.split(",")]
        else:
            fault_ranks = [int(force_ranks)]
        assert all(0 <= r < world_size for r in fault_ranks), \
            f"Fault ranks must be between 0 and {world_size - 1}"
        assert len(fault_ranks) > 0, "Must specify at least one fault rank"
    else:
        assert cli_args.fault_injector_num_ranks is not None, "Must specify either force_ranks or num_ranks"
        fault_ranks = rng.sample(range(1, world_size), k=cli_args.fault_injector_num_ranks)

    return fault_ranks


def get_fault(cli_args):
    global rng

    fault_types = cli_args.fault_injector_fault_types
    fault_probabilities = cli_args.fault_injector_fault_probabilities

    if ',' in fault_types:
        fault_types = [Fault[t.upper()] for t in fault_types.split(",")]
    else:
        fault_types = [Fault[fault_types.upper()]]

    if fault_probabilities is not None:
        if ',' in fault_probabilities:
            fault_probabilities = [float(p) for p in fault_probabilities.split(",")]
        else:
            fault_probabilities = [float(fault_probabilities)]
        fault_probabilities = [p / sum(fault_probabilities) for p in fault_probabilities]
    else:
        fault_probabilities = [1 / len(fault_types) for _ in fault_types]

    assert len(fault_types) > 0, "Must specify at least one fault type"
    assert len(fault_types) == len(fault_probabilities), \
        "Number of fault types and fault probabilities must match"

    return rng.choices(fault_types, fault_probabilities, k=1)[0]


def get_fault_delay(cli_args):
    global rng

    if cli_args.fault_injector_fault_iteration is not None:
        assert cli_args.fault_injector_fault_delay is None and cli_args.fault_injector_mtti_seconds is None, \
            "fault_injector_fault_iteration is mutually exclusive with fault_delay and mtti_seconds"
        return 0.0

    fault_delay = cli_args.fault_injector_fault_delay
    if fault_delay is None:
        mtti_seconds = cli_args.fault_injector_mtti_seconds
        offset_seconds = cli_args.fault_injector_offset_seconds
        lambda_inj = 1.0 / mtti_seconds
        fault_delay = offset_seconds + (-math.log(1.0 - rng.random()) / lambda_inj)

    return fault_delay


def setup_fault_injection(cli_args):
    global rng

    my_rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device("cuda", torch.cuda.current_device())
    plan_tensor = torch.zeros(world_size + 1, device=device)

    clear_workload_exception()

    if my_rank == 0:
        if rng is None:
            rng = random.Random(cli_args.fault_injector_seed)

        fault_ranks = get_fault_ranks(cli_args)
        fault = get_fault(cli_args)
        fault_delay = get_fault_delay(cli_args)

        for rank in fault_ranks:
            plan_tensor[rank] = fault.value
        plan_tensor[world_size] = fault_delay

    dist.broadcast(plan_tensor, src=0)

    is_target_rank = bool(plan_tensor[my_rank].item())

    if is_target_rank:
        fault = Fault(int(plan_tensor[my_rank].item()))
        fault_delay = float(plan_tensor[world_size].item())
        current_time = datetime.datetime.now()
        fault_time = current_time + datetime.timedelta(seconds=fault_delay)
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")
        fault_timestamp = fault_time.strftime("%Y-%m-%d %H:%M:%S.%f")
        logger.warning(
            f"[{timestamp}] FAULT INJECTION: Rank {my_rank} will inject fault "
            f"{fault.name} at {fault_timestamp}"
        )
        dispatch_fault_injection(fault=fault, delay=fault_delay, callback=None)
