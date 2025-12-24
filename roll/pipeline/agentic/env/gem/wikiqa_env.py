import random
from typing import Tuple, Any, SupportsFloat, Optional, Dict

from datasets import Dataset
from gem.envs.qa_env import QaEnv as GEMQaEnv
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE
from gem.utils.parsing import extract_last_boxed_answer, extract_last_tagged_answer
from roll.datasets.global_dataset import GlobalDataset, GlobalDatasetManager
from roll.utils.constants import RAY_NAMESPACE
import ray
from functools import partial

# def apply_prompt(example, question_key: str = "question"):
#     prompt_template = (
#         "For any question, always reason through your thought process using:\n"
#         "<think> your reasoning here </think>\n"
#         "Then, provide your final answer using:\n"
#         "<answer> your answer here </answer>\n\n"
#         "Question: {question}\n"
#     )
#     example[question_key] = prompt_template.format(question=example[question_key])
#     return example

class QaEnv(GEMQaEnv):
    def __init__(
        self,
        dataset_name: Optional[str] = "",
        split: Optional[str] = None,
        dataset: Optional[Dataset] = None,
        id_key: str = "id",
        question_key: str = "question",
        answer_key: str = "answer",
        seed: int = 0,
        extract_boxed: bool = False,
        mode: str = "train",
        load_from_cache_file: bool = False,  # False to force re-run the apply_prompt_func, useful when apply_prompt is changed
        **_,
    ):
        Env.__init__(self)
        self.seed = seed
        self.id_key = id_key
        self.question_key = question_key
        self.answer_key = answer_key
        self.mode = mode

        # Convert train/val mode to sample/traversal for GlobalDataset
        global_dataset_mode = "sample" if self.mode == "train" else "traversal"
        self.dataset = GlobalDataset.options(name=f"{self.mode}_{dataset_name}",
                                             get_if_exists=True,
                                             namespace=RAY_NAMESPACE).remote(dataset_name=dataset_name,
                                                                              split=split,
                                                                             mode=global_dataset_mode)
        
        self.dataset_manager = GlobalDatasetManager.options(name=f"{self.mode}_dataset_manager",
                                                            get_if_exists=True,
                                                            namespace=RAY_NAMESPACE).remote()
        ray.get(self.dataset_manager.register.remote(dataset_name=dataset_name, dataset_ref=self.dataset))

        # apply_prompt_func = partial(apply_prompt, question_key=question_key)
        # ray.get(self.dataset.process.remote(apply_prompt_func, load_from_cache_file))

        self.idx = 0
        self.epoch = 0

        if extract_boxed:
            self.extractor = extract_last_boxed_answer
        else:
            self.extractor = extract_last_tagged_answer

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        model_answer = self.extractor(action)
        action_is_valid = True
        if model_answer is None:
            reward = 0.0
            action_is_valid = False
        else:
            is_correct = self.check_correct(model_answer, self.answer)
            reward = 1.0 if is_correct else 0.0
        metrics = {
            "action_is_valid": action_is_valid,
            "success": reward > 0,
            "raw_reward": reward,
        }
        metrics_agg_mode = {
            "action_is_valid": "mean",
            "success": "last",
            "raw_reward": "last",
        }
        info = {
            "metrics": metrics,
            "metrics_agg_mode": metrics_agg_mode,
            "model_answer": model_answer if model_answer is not None else ""
        }
        return TERMINAL_STATE, reward, True, True, info

    def reset(self, seed: Optional[None] = None) -> Tuple[str, dict[str, Any]]:
        """Sample a question from the dataset."""
        Env.reset(self, seed)
        data: Optional[Dict] = ray.get(self.dataset.get_data_item.remote(seed=seed))
        if data is None:
            return None, None
        self.first_obs = data[self.question_key]
        self.answer = data[self.answer_key]
        self.idx += 1
        example_id = ""
        if self.id_key in data:
            example_id = data[self.id_key]
        return self.first_obs, {"example_id": example_id, "answer": self.answer}