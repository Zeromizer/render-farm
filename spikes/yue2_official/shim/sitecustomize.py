"""Loaded only for the official skill's child processes when YUE2_ATTENTION is set.

yue2-infer 0.1.5's GraphAR picks attention "auto": flash if the aten FlashAttention op is
REGISTERED, else cuDNN, else public SDPA. Windows torch 2.10.0+cu128 registers the op but is
built without USE_FLASH_ATTENTION, so "auto" picks flash and the first decode step raises.
This sets the default to the backend run_official.py's probe found working, in the authors'
own preference order (flash -> cudnn -> sdpa). No other behaviour changes."""
import os

_backend = os.environ.get("YUE2_ATTENTION")
if _backend:
    try:
        import functools
        import yue2.cuda_graph as _cg

        _orig = _cg.GraphAR.__init__

        @functools.wraps(_orig)
        def _init(self, *args, attention_backend="auto", **kwargs):
            if attention_backend == "auto":
                attention_backend = _backend
            return _orig(self, *args, attention_backend=attention_backend, **kwargs)

        _cg.GraphAR.__init__ = _init
    except ImportError:
        pass  # CPU-only helpers run without the runtime installed
