import json
import os.path
from itertools import count
from typing import Any

import ray
import torch
from codetiming import Timer

from roll.distributed.scheduler.rollout_scheduler import RolloutScheduler
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.pipeline.agentic.utils import dump_rollout_trajectories
from roll.pipeline.base_pipeline import BasePipeline
from roll.utils.functionals import (
    reduce_metrics,
)
from roll.utils.logging import get_logger
from datetime import datetime
import time

logger = get_logger()


class AgenticRolloutPipeline(BasePipeline):
    """
    this is just for env rollout
    """
    def __init__(self, pipeline_config: AgenticConfig):
        super().__init__(pipeline_config)
        self.pipeline_config: AgenticConfig

        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)
        self.use_policy_model = self.pipeline_config.train_env_manager.llm_proxy.proxy_type == "policy"

        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        self.download_models(self.actor_infer)
        self.tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.actor_train.model_args)

        self.rollout_scheduler = ray.remote(RolloutScheduler).remote(
            config=self.pipeline_config,
            env_manager_config=self.pipeline_config.train_env_manager,
            resource_manager=self.resource_manager,
            infer_cluster=self.actor_infer,
            mode="val",
        )

        if self.use_policy_model:
            self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=True)

    @torch.no_grad()
    def run(self):
        start_time = time.time()  # 时间戳格式
        start_time_str = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d-%H%M%S')  # 格式化时间字符串

        for global_step in (count() if self.pipeline_config.max_steps == -1 else range(self.pipeline_config.max_steps)):
            logger.info(f"pipeline rollout global step {global_step} start...")
            metrics = {}
            batch: DataProto = DataProto()
            batch.meta_info = {"global_step": global_step}

            with Timer(name="rollout", logger=None) as rollout_timer:
                if self.use_policy_model:
                    batch.meta_info["is_offload_states"] = True
                    self.actor_infer.start_server(data=batch)
                batch = ray.get(self.rollout_scheduler.get_batch.remote(batch, self.pipeline_config.rollout_batch_size))
                if batch is None:
                    break

            metrics["time/rollout"] = rollout_timer.last
            eval_metrics = reduce_metrics(batch.meta_info.get("metrics", {}))
            eval_score = batch.batch["scores"].sum(-1)
            eval_metrics["score/mean"] = torch.mean(eval_score).detach().item()
            eval_metrics["score/max"] = torch.max(eval_score).detach().item()
            eval_metrics["score/min"] = torch.min(eval_score).detach().item()

            batch_grouped = batch.group_by(keys="tags")
            for group_name, group_batch in batch_grouped.items():
                eval_score = group_batch.batch["scores"].sum(-1)
                eval_metrics[f"{group_name}/score/mean"] = torch.mean(eval_score).detach().item()
                eval_metrics[f"{group_name}/score/max"] = torch.max(eval_score).detach().item()
                eval_metrics[f"{group_name}/score/min"] = torch.min(eval_score).detach().item()
                group_eval_metrics = reduce_metrics(group_batch.meta_info.get("metrics", {}))
                eval_metrics.update({f"{group_name}/{k}": v for k, v in group_eval_metrics.items()})

            metrics.update({f"val/{k}": v for k, v in eval_metrics.items()})
            batch.meta_info["global_step"] = global_step
            metrics["system/samples"] = (global_step + 1) * batch.batch.shape[0]

            self.tracker.log(values=metrics, step=global_step)

            dump_rollout_trajectories(self.pipeline_config.rollout_dump_dir, global_step, batch)

            if global_step % self.pipeline_config.logging_steps == 0:
                if int(os.environ.get("RAY_PROFILING", "0")):
                    timeline_dir = os.path.join(self.pipeline_config.profiler_output_dir, "timeline")
                    os.makedirs(timeline_dir, exist_ok=True)
                    ray.timeline(
                        filename=os.path.join(timeline_dir, f"timeline-step-{global_step}.json"),
                    )

                log_res = []
                batch_grouped = batch.group_by(keys="traj_id")
                for group_name, group_batch in batch_grouped.items():
                    prompt_mask = group_batch.batch["prompt_mask"]
                    non_prompt_mask = torch.logical_not(group_batch.batch["prompt_mask"]) * group_batch.batch["attention_mask"]
                    input_ids = group_batch.batch["input_ids"]
                    prompt_ids_list = [input_ids[i][mask.bool()] for i, mask in enumerate(prompt_mask)]
                    response_ids_list = [input_ids[i][mask.bool()] for i, mask in enumerate(non_prompt_mask)]
                    prompts = self.tokenizer.batch_decode(prompt_ids_list, skip_special_tokens=False)
                    responses = self.tokenizer.batch_decode(response_ids_list, skip_special_tokens=False)
                    episode_scores = group_batch.non_tensor_batch["episode_scores"].tolist()
                    step_scores = group_batch.non_tensor_batch["step_scores"].tolist()
                    example_ids = group_batch.non_tensor_batch["example_ids"].tolist()
                    answers = group_batch.non_tensor_batch["answers"].tolist()
                    model_answers = group_batch.non_tensor_batch["model_answers"].tolist()
                    traj_ids = group_batch.non_tensor_batch["traj_id"]
                    traj_group_ids = group_batch.non_tensor_batch["traj_group_id"]
                    if not isinstance(step_scores[0], float):
                        step_scores = [t.tolist() for t in step_scores]

                    for example_id, prompt, response, answer, model_answer, episode_score, step_score, traj_id, traj_group_id in zip(
                            example_ids, prompts, responses, answers, model_answers, episode_scores, step_scores, traj_ids, traj_group_ids
                    ):
                        log_res.append(
                            {
                                "example_id": example_id,
                                "prompt": prompt,
                                "response": response,
                                "answer": answer,
                                "model_answer": model_answer,
                                "episode_score": episode_score,
                                "step_score": step_score,
                                "traj_id": traj_id,
                                "traj_group_id": traj_group_id,
                            }
                        )
                try:
                    self.save_to_jsonl(log_res, '/home/chenkaiyuan/ROLL_Rollout/agentic_val_rollout/'+start_time_str, f"rollout-{global_step}.json")
                except Exception as e:
                    logger.error(f"Error saving rollout result: {e}")
                
                logger.info(json.dumps(log_res[:10], ensure_ascii=False))
                logger.info(json.dumps(metrics, ensure_ascii=False))

            logger.info(f"pipeline step {global_step} finished")
            global_step += 1
        ray.get(self.rollout_scheduler.shutdown.remote())
        logger.info("pipeline complete!")

    def save_to_jsonl(self, data, folder_path, file_name):
        # 1. 检查文件夹是否存在
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  # 创建文件夹（递归创建）

        # 2. 构造完整文件路径
        file_path = os.path.join(folder_path, file_name)

        # 3. 保存数据到 JSONL 文件
        with open(file_path, 'w', encoding='utf-8') as f:
            if isinstance(data, list):  # 如果数据是列表，逐行写入每个字典
                for item in data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')  # 每个 JSON 对象占一行
            elif isinstance(data, dict):  # 如果数据是单个字典，直接写入一行
                json.dump(data, f, ensure_ascii=False)
                f.write('\n')
            else:
                raise ValueError("数据必须是字典或列表类型")
        
        print(f"Rollout result is saved to {file_path}..")
