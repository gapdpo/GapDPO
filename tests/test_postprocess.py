import unittest

from madb.postprocess import clean_model_utterance


class PostprocessTest(unittest.TestCase):
    def test_removes_think_block_and_role_prefix(self):
        text = "<think>internal reasoning</think>\n공격자: \"지금 바로 확인이 필요해.\""

        self.assertEqual(clean_model_utterance(text), "지금 바로 확인이 필요해.")

    def test_removes_stage_direction_but_keeps_masked_placeholder(self):
        text = "[딥보이스 스타일의 음성 변조] [MASKED_ACCOUNT]로 정산 확인이 필요합니다."

        self.assertEqual(clean_model_utterance(text), "[MASKED_ACCOUNT]로 정산 확인이 필요합니다.")


if __name__ == "__main__":
    unittest.main()
