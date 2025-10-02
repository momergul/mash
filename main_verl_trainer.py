# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
from verl_training.trainer.agentic_trainer import AgenticGRPOTrainer 
import os
import ray
import hydra
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["VLLM_LOGGING_LEVEL"] = "WARN"

@hydra.main(config_path='verl_training', config_name='agentic_grpo_trainer', version_base=None)
def main(config):
    run_ppo(config)

def run_ppo(config) -> None:
    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init()

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:

    def define_workers(self, config):
        # define worker classes
        assert config.actor_rollout_ref.actor.strategy == 'fsdp'
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl_training.workers.fsdp_workers import AgenticActorRolloutRefWorker
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        ray_worker_group_cls = RayWorkerGroup
        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        # Define the Actor, Rollout and Reference classes
        role_worker_mapping = {
            Role.ActorRollout : ray.remote(AgenticActorRolloutRefWorker),
            Role.RefPolicy : ray.remote(AgenticActorRolloutRefWorker)                
        }

        # Define the resource pool manager
        mapping = {role : global_pool_id for role in role_worker_mapping}
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        return role_worker_mapping, resource_pool_manager, ray_worker_group_cls

    def get_rule_based_rewards(self, tokenizer, config):
        if config.reward_model.use_neural_reward:
            return None, None

        from verl_training.rule_based_rewards.reward_manager import QARewardManager
        reward_fn = QARewardManager(config.reward_model.rule_based_reward)
        val_reward_fn = QARewardManager(config.reward_model.rule_based_reward)

        return reward_fn, val_reward_fn

    def set_datapaths(self, config):
        import os
        parent_dir = "data"
        if config.data.task == "biography":
            config.data.train_files = os.path.join(parent_dir, "entities", "train_verl.parquet")
            config.data.val_files = os.path.join(parent_dir, "entities", "val_verl.parquet")
            config.data.test_files = os.path.join(parent_dir, "entities", "test_verl.parquet")
        elif config.data.task == "shortform":
            dataset_prefix = config.data.shortform_dataset_prefix
            config.data.train_files = os.path.join(parent_dir, "shortform_qa", f"{dataset_prefix}_train.parquet")
            config.data.val_files = os.path.join(parent_dir, "shortform_qa", f"{dataset_prefix}_fast_dev.parquet")
            config.data.test_files = os.path.join(parent_dir, "shortform_qa", f"{dataset_prefix}_test.parquet")
            if config.trainer.sft_subsample:
                prefix = config.data.shortform_prompt_type
                config.data.test_files = os.path.join(parent_dir, "shortform_qa", f"{prefix}_prompt_{dataset_prefix}_sft_subsample.parquet") 
    def set_model_path(self, config):
        import os
        if config.actor_rollout_ref.model.load_from_hf:
            model_name = config.actor_rollout_ref.model.hf_model
            if "naive_sft" in model_name:
                path = f"momergul/{model_name}"                
            elif "qwen" in model_name:
                path = "Qwen/Qwen2.5-3B"        
                if 'instruct' in model_name:
                    path += '-Instruct'
        else:
            experiment_dir = config.trainer.experiment_dir
            path = os.path.join(
                experiment_dir,
                config.actor_rollout_ref.model.load_experiment_folder,
                config.actor_rollout_ref.model.load_experiment_name,
                config.actor_rollout_ref.model.load_experiment_checkpoint
            )

        config.actor_rollout_ref.model.path = path

    def run(self, config):
        from verl.utils.fs import copy_to_local
        # print initial config
        from pprint import pprint
        from omegaconf import OmegaConf
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # Define train/val data paths
        self.set_datapaths(config)
        print(config.data.train_files)

        # download the checkpoint from hdfs
        self.set_model_path(config)
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        print(local_path)

        # instantiate tokenizer
        from verl.utils import hf_tokenizer, hf_processor
        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes and reward functions
        role_worker_mapping, resource_pool_manager, ray_worker_group_cls = self.define_workers(config)
        reward_fn, val_reward_fn = self.get_rule_based_rewards(tokenizer, config) 

        trainer = AgenticGRPOTrainer(config=config,
                                     tokenizer=tokenizer,
                                     processor=processor,
                                     role_worker_mapping=role_worker_mapping,
                                     resource_pool_manager=resource_pool_manager,
                                     ray_worker_group_cls=ray_worker_group_cls,
                                     reward_fn=reward_fn,
                                     val_reward_fn=val_reward_fn)
        print("Passed trainer init")
        trainer.init_workers()
        print("Passed worker init")

        if config.trainer.pure_test:
            test_metric = config.trainer.test_load_metric
            prefix = "test_outputs" if config.trainer.pure_test_suffix == '' else f'test_outputs_{config.trainer.pure_test_suffix}'
            trainer.pure_test(f"{prefix}_{test_metric}", dataloader=trainer.test_dataloader)
        else:
            trainer.fit() # TODO
            if not config.trainer.skip_final_test:
                for metric in config.trainer.save_metrics:
                    config.trainer.test_load_metric = metric
                    trainer.pure_test(f'test_outputs_{metric}', dataloader=trainer.test_dataloader)


if __name__ == '__main__':
    main()
