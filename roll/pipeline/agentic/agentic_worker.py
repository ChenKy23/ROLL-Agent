import torch
from roll.distributed.scheduler.protocol import DataProto
from roll.utils.functionals import (
    masked_mean,
    compute_approx_kl,
    segment_sum,
    segment_lengths_by_last_one,
    trim_trailing_zeros,
    agg_loss,
)

from roll.pipeline.base_worker import ActorWorker as BaseActorWorker
from roll.utils.logging import get_logger

logger = get_logger()

class ActorWorker(BaseActorWorker):
    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        """
        loss func接口定义:
            data: DataProto, 由train_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """

        response_mask = data.batch["response_mask"][:, 1:].long()
        ref_log_probs = data.batch["ref_log_probs"]
        old_log_probs = data.batch["old_log_probs"]
        advantages = data.batch["advantages"]
        step_diff_pos_weights = data.batch["step_diff_pos_weights"].detach()
        step_ref_neg_weights = data.batch["step_ref_neg_weights"].detach()
        response_step_lengths = data.batch["response_step_lengths"]
        step_lengths = data.batch["step_lengths"]
        step_lengths[:, 0] -= 1
        step_lengths = trim_trailing_zeros(step_lengths)
        grouped_difficulty = data.batch["grouped_difficulty"]
        logger.info(f"grouped_difficulty: {grouped_difficulty.tolist()}")
        # logger.info(f"step_lengths: {step_lengths}")
        logger.info(f"step_diff_pos_weights: {step_diff_pos_weights.tolist()}")
        logger.info(f"step_ref_neg_weights: {step_ref_neg_weights.tolist()}")

        total_scores = data.batch["scores"][:, 1:].long().sum(-1)
        logger.info(f"total_scores: {total_scores.tolist()}")

        # logger.info(f"org response_step_lengths: {segment_sum(response_mask, step_lengths).tolist()}")
        # logger.info(f"data response_step_lengths: {data.batch['response_step_lengths'].tolist()}")

        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor, input_ids=data.batch["input_ids"], attention_mask=data.batch["response_mask"]
        )

        ratio = (log_probs - old_log_probs).exp()

        pg_clip_low = self.pipeline_config.pg_clip_low if self.pipeline_config.use_pg_clip_range else self.pipeline_config.pg_clip
        pg_clip_high = self.pipeline_config.pg_clip_high if self.pipeline_config.use_pg_clip_range else self.pipeline_config.pg_clip  
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - pg_clip_low, 1 + pg_clip_high) * advantages
        pg_loss = -torch.min(surr1, surr2)
        if self.pipeline_config.dual_clip_loss:
            dual_clip_loss = -torch.max(-pg_loss, (1 + self.pipeline_config.pg_clip * 2) * advantages)
            pg_loss = torch.where(advantages < 0, dual_clip_loss, pg_loss)

        step_pg_loss = segment_sum(pg_loss * response_mask, step_lengths)
        logger.info(f"before step_pg_loss: {step_pg_loss.tolist()}")
        _, max_step_len = step_pg_loss.shape
        response_step_lengths = response_step_lengths[:, :max_step_len]
        step_pg_loss = (step_pg_loss / response_step_lengths.clamp(min=1.0)) * (response_step_lengths != 0).to(step_pg_loss.dtype)
        logger.info(f"after step_pg_loss: {step_pg_loss.tolist()}")
        step_total_loss = torch.tensor(0.0, device=pg_loss.device, dtype=pg_loss.dtype)
        if self.pipeline_config.use_positive_weight and bool(torch.any(step_diff_pos_weights)):
            step_diff_pos_weights = step_diff_pos_weights[:, :max_step_len]
            logger.info(f"cliped step_diff_pos_weights: {step_diff_pos_weights.tolist()}")
            if self.pipeline_config.use_difficulty_weight:
                pos_grouped_difficulty = 2.0 - grouped_difficulty
                # pos_grouped_difficulty = torch.where(pos_grouped_difficulty == 0, 0.1, pos_grouped_difficulty)
                logger.info(f"pos_grouped_difficulty: {pos_grouped_difficulty.tolist()}")
                step_diff_pos_weights = pos_grouped_difficulty * step_diff_pos_weights
                logger.info(f"weighted step_diff_pos_weights: {step_diff_pos_weights.tolist()}")
            logger.info(f"bool step_diff_pos_weights: {(step_diff_pos_weights > 0).float().tolist()}")
            step_pos_pg_loss = agg_loss(loss_mat=step_pg_loss*step_diff_pos_weights, loss_mask=(step_diff_pos_weights > 0).float(), loss_agg_mode=self.pipeline_config.loss_agg_mode)
            logger.info(f"step_pos_pg_loss: {step_pos_pg_loss}")
            step_total_loss = step_total_loss + step_pos_pg_loss

        if self.pipeline_config.use_negative_weight and bool(torch.any(step_ref_neg_weights)):
            step_ref_neg_weights = step_ref_neg_weights[:, :max_step_len]
            logger.info(f"cliped step_ref_neg_weights: {step_ref_neg_weights.tolist()}")
            if self.pipeline_config.use_difficulty_weight:
                # neg_grouped_difficulty = torch.where(grouped_difficulty == 0, 0.1, grouped_difficulty)
                neg_grouped_difficulty = grouped_difficulty.clone()
                logger.info(f"neg_grouped_difficulty: {neg_grouped_difficulty.tolist()}")
                step_ref_neg_weights = neg_grouped_difficulty * step_ref_neg_weights
                logger.info(f"weighted step_ref_neg_weights: {step_ref_neg_weights.tolist()}")
            logger.info(f"bool step_ref_neg_weights: {(step_ref_neg_weights > 0).float().tolist()}")
            step_neg_pg_loss = agg_loss(loss_mat=step_pg_loss*step_ref_neg_weights, loss_mask=(step_ref_neg_weights > 0).float(), loss_agg_mode=self.pipeline_config.loss_agg_mode)
            logger.info(f"step_neg_pg_loss: {step_neg_pg_loss}")
            step_total_loss = step_total_loss + step_neg_pg_loss

        logger.info(f"step_total_loss: {step_total_loss}")
        if self.pipeline_config.use_weighted_loss_only:
            pg_loss = step_total_loss
        else:
            pg_loss = agg_loss(loss_mat=pg_loss, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode)
            logger.info(f"before pg_loss: {pg_loss}")
            pg_loss = pg_loss + step_total_loss * self.pipeline_config.step_loss_coef

        logger.info(f"after pg_loss: {pg_loss}")
        kl_loss = compute_approx_kl(log_probs=log_probs, log_probs_base=ref_log_probs, action_mask=response_mask,
                                    kl_penalty="k3")
        kl_loss = agg_loss(loss_mat=kl_loss, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode)

        approxkl = compute_approx_kl(
            log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="mse"
        )
        policykl = compute_approx_kl(
            log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="kl"
        )
        clipped_low = (ratio < 1 - pg_clip_low).float()
        clipped_high = (ratio > 1 + pg_clip_high).float()
        clipped = (clipped_low + clipped_high).float()

        if self.pipeline_config.use_kl_loss:
            total_loss = pg_loss + kl_loss * self.pipeline_config.kl_loss_coef
        else:
            total_loss = pg_loss
        if self.pipeline_config.entropy_loss_coef > 0:
            entropy = self.strategy.op_compute_entropy(logits=output_tensor, attention_mask=data.batch["response_mask"])
            entropy_loss = agg_loss(
                loss_mat=entropy,
                loss_mask=response_mask,
                loss_agg_mode=self.pipeline_config.loss_agg_mode,
            )
            logger.info(f"entropy_loss: {entropy_loss}")
            total_loss = total_loss - entropy_loss * self.pipeline_config.entropy_loss_coef
        logger.info(f"total_loss: {total_loss}")

        pg_metrics = {
            "actor/ppo_ratio_high_clipfrac": clipped_high.mean().detach().item(),
            "actor/ppo_ratio_low_clipfrac": clipped_low.mean().detach().item(),
            "actor/ppo_ratio_clipfrac": clipped.mean().detach().item(),
            "actor/ratio_mean": masked_mean(ratio, response_mask, dim=-1).mean().detach().item(),
            "actor/ratio_max": torch.max(ratio * response_mask).detach().item(),
            "actor/ratio_min": torch.min(ratio * response_mask + (1 - response_mask) * 1e10).detach().item(),
            "actor/clipfrac": agg_loss(loss_mat=torch.lt(surr2, surr1).float(), loss_mask=response_mask,
                                       loss_agg_mode=self.pipeline_config.loss_agg_mode).detach().item(),
            "actor/pg_loss": pg_loss.detach().item(),
            "actor/kl_loss": kl_loss.detach().item(),
            "actor/total_loss": total_loss.detach().item(),
            "actor/approxkl": agg_loss(loss_mat=approxkl, loss_mask=response_mask,
                                       loss_agg_mode=self.pipeline_config.loss_agg_mode).detach().item(),
            "actor/policykl": agg_loss(loss_mat=policykl, loss_mask=response_mask,
                                       loss_agg_mode=self.pipeline_config.loss_agg_mode).detach().item(),
            "actor/step_pos_pg_loss": step_pos_pg_loss.detach().item() if self.pipeline_config.use_positive_weight and bool(torch.any(step_diff_pos_weights)) else 0.0,
            "actor/step_neg_pg_loss": step_neg_pg_loss.detach().item() if self.pipeline_config.use_negative_weight and bool(torch.any(step_ref_neg_weights)) else 0.0,
            "actor/grouped_difficulty": grouped_difficulty.mean().detach().item() if self.pipeline_config.use_difficulty_weight else 0.0,
            "actor/pos_grouped_difficulty": pos_grouped_difficulty.mean().detach().item() if self.pipeline_config.use_positive_weight and self.pipeline_config.use_difficulty_weight and bool(torch.any(step_diff_pos_weights)) else 0.0,
            "actor/neg_grouped_difficulty": neg_grouped_difficulty.mean().detach().item() if self.pipeline_config.use_negative_weight and self.pipeline_config.use_difficulty_weight and bool(torch.any(step_ref_neg_weights)) else 0.0,
        }

        return total_loss, pg_metrics