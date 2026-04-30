"""
reencode_patch.py — ReEncode tool and fallback ReasoningConditioner.

ReEncode: a tool the model can explicitly call to request a re-encode of a
specific image through the reasoning-conditioned ViT.  When the full
ReasoningConditionerV2 is live (--use_conditioner), the conditioner also runs
automatically after every crop/zoom; this explicit tool is an alternative
entry point for models trained with a vcot prompt.

ReasoningConditioner: lightweight fallback when --use_conditioner is not set.
It is a no-op identity so the pipeline works without the conditioner.
"""

from qwen_agent.tools.base import BaseTool, register_tool


@register_tool("reencode")
class ReEncode(BaseTool):
    """Re-encode a specific image through the reasoning-conditioned visual encoder."""

    @property
    def description(self):
        return (
            "Re-encode a specific image through the reasoning-conditioned visual "
            "encoder to obtain updated visual features aligned with the current "
            "reasoning context."
        )

    parameters = {
        "type": "object",
        "properties": {
            "target_image": {
                "type": "number",
                "description": (
                    "The index of the image to re-encode. "
                    "Index from 1 to the number of images. Choose 1 for the original image."
                ),
            }
        },
        "required": ["target_image"],
    }

    def call(self, image):
        # Identity: the actual conditioning is applied by the experience_maker
        # after execute_tool returns, using self.conditioner.process().
        return image


class ReasoningConditioner:
    """Heuristic-only fallback conditioner (no ViT conditioning).

    Used when --use_conditioner is not set.  All methods are no-ops so the
    pipeline behaves identically to the original Pixel-Reasoner.
    """

    def process(self, image, focus_hint="", reasoning_history=""):
        return image

    def get_trainable_parameters(self):
        return []
