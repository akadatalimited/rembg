import os
from typing import Type

import onnxruntime as ort

from .sessions import sessions_class
from .sessions.base import BaseSession
from .sessions.u2net import U2netSession


def new_session(
    model_name: str = "birefnet-massive", providers=None, *args, **kwargs
) -> BaseSession:
    """
    Create a new session object based on the specified model name.
    """
    session_class: Type[BaseSession] = U2netSession

    for sc in sessions_class:
        if sc.name() == model_name:
            session_class = sc
            break

    # Configure session options for high parallelism and GPU prioritization
    sess_opts = ort.SessionOptions()
    if "OMP_NUM_THREADS" in os.environ:
        sess_opts.inter_op_num_threads = int(os.environ["OMP_NUM_THREADS"])
        sess_opts.intra_op_num_threads = int(os.environ["OMP_NUM_THREADS"])
    sess_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL  # Set parallel execution mode
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # Maximize graph optimization

    # Ensure CUDA provider is prioritized if available
    if not providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    return session_class(model_name, sess_opts, providers, *args, **kwargs)
