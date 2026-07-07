"""(已搬家) 架構登記表改用時間軸版本檔:見 `trpg/rl/architectures/`。

一檔一版本、按時間排序、各有代號、可回退。此檔僅為向後相容 re-export;
新程式請直接 `from trpg.rl.architectures import build, load_net, save_net, timeline`。
"""
from .architectures import (  # noqa: F401
    REGISTRY, TIMELINE, build, timeline,
    save_net, load_net, write_sidecar, read_codename,
)
