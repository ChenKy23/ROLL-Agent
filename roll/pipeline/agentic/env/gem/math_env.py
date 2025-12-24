import logging
import multiprocessing
import random
from typing import Optional, Tuple, Any, SupportsFloat, Dict

from datasets import load_dataset, Dataset, DatasetDict
from gem import Env
from gem.envs.math_env import MathEnv as GEMMathEnv
from gem.utils.constants import TERMINAL_STATE
from gem.utils.parsing import extract_last_boxed_answer
import ray

from roll.datasets.global_dataset import GlobalDataset, GlobalDatasetManager
from roll.utils.constants import RAY_NAMESPACE

from math_verify import parse, verify

logger = logging.getLogger(__name__)

class MathEnv(GEMMathEnv):

    def __init__(
            self,
            dataset_name: Optional[str] = "",
            split: Optional[str] = None,
            dataset: Optional[Dataset] = None,
            id_key: str = "id",
            question_key: str = "problem",
            answer_key: str = "answer",
            seed: int = 0,
            mode: str = "train",
            **_,
    ):
        Env.__init__(self)
        self.seed = seed
        self.question_key = question_key
        self.answer_key = answer_key
        self.id_key = id_key
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
        self.idx = 0
        self.epoch = 0
        # Process pool is used to enable the timeout mechanism for answer grading in a potential distributed training setup
        self.mp_pool = multiprocessing.Pool(1)

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
        return self.first_obs, {"env_instruction": "", "example_id": example_id, "answer": self.answer}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        model_answer = extract_last_boxed_answer(action)
        action_is_valid = True
        if model_answer is None:
            reward = 0
            action_is_valid = False
        else:
            res = self.mp_pool.apply_async(
                self.robust_check_correct, (model_answer, self.answer)
            )
            try:
                is_correct = res.get(timeout=1)
            except (multiprocessing.context.TimeoutError, Exception):
                is_correct = False
            reward = 1.0 if is_correct else 0

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
    
    @staticmethod
    def robust_check_correct(model_answer: str, gt_answer: str) -> bool:
        """Check if the action is correct."""
        if not model_answer.startswith("\\boxed{"):
            model_answer = "\\boxed{" + model_answer + "}"
        if not gt_answer.startswith("\\boxed{"):
            gt_answer = "\\boxed{" + gt_answer + "}"
        # parse with math_verify
        model_answer = parse(model_answer)

        # get correct answers from the dataset entry
        if isinstance(gt_answer, (str, float, int)):
            correct_answers = [str(gt_answer)]
        elif isinstance(gt_answer, list):
            correct_answers = gt_answer
        else:
            raise ValueError(f"Unexpected answer type: {type(gt_answer)}")

        # check against all possible correct answers
        # (math_verify.parse handles extraction e.g. from \\boxed{...})
        is_correct = False
        for correct_answer in correct_answers:
            correct_answer = parse(str(correct_answer))
            if verify(correct_answer, model_answer):
                is_correct = True
                break
        return is_correct