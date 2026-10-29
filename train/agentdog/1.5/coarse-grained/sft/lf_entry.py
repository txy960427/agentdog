"""保留官方Qwen聊天模板，只在本次训练进程对齐其user正文strip行为。"""

from copy import deepcopy

from llamafactory.data.formatter import StringFormatter


class OfficialUserFormatter(StringFormatter):
    def apply(self, **kwargs):
        # Qwen官方Jinja对user文本做strip；默认LF格式器不做，末尾会差一个换行token。
        return super().apply(**{**kwargs, "content": kwargs["content"].strip()})


def setup_template():
    from llamafactory.data.template import TEMPLATES

    template = deepcopy(TEMPLATES["qwen3_5"])
    template.format_user = OfficialUserFormatter(slots=template.format_user.slots)
    template.replace_jinja_template = False  # 保存模型时继续使用底座官方Jinja，保持推理一致。
    TEMPLATES["qwen3_5"] = template


if __name__ == "__main__":
    setup_template()
    from llamafactory.train.tuner import run_exp

    run_exp()
