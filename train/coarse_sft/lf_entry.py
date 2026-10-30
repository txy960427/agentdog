"""保留官方Qwen聊天模板，只在本次训练进程对齐其user正文strip行为。"""

from __future__ import annotations

import os
from copy import deepcopy

from llamafactory.data.formatter import StringFormatter
from transformers import TrainerCallback


class OfficialUserFormatter(StringFormatter):
    def apply(self, **kwargs):
        # Qwen官方Jinja对user文本做strip；默认LF格式器不做，末尾会差一个换行token。
        return super().apply(**{**kwargs, "content": kwargs["content"].strip()})


class SkipEarlySaveCallback(TrainerCallback):
    """前 save_after_steps 个优化 step 不写 checkpoint；之后仍按 save_steps。"""

    def __init__(self, save_after_steps: int) -> None:
        self.save_after_steps = max(0, int(save_after_steps))

    def on_step_end(self, args, state, control, **kwargs):
        if self.save_after_steps and state.global_step < self.save_after_steps:
            control.should_save = False
        return control


def setup_template():
    from llamafactory.data.template import TEMPLATES

    template = deepcopy(TEMPLATES["qwen3_5"])
    template.format_user = OfficialUserFormatter(slots=template.format_user.slots)
    template.replace_jinja_template = False  # 保存模型时继续使用底座官方Jinja，保持推理一致。
    TEMPLATES["qwen3_5"] = template


if __name__ == "__main__":
    setup_template()
    from llamafactory.train.tuner import run_exp

    save_after = int(os.environ.get("SAVE_AFTER_STEPS", "0") or 0)
    callbacks = [SkipEarlySaveCallback(save_after)] if save_after > 0 else None
    run_exp(callbacks=callbacks)
