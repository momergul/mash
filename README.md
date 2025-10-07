Pay-Per-Search Models are Abstention Models
===========================================

This is the code repository for the paper "[Pay-Per-Search Models are Abstention Models](https://arxiv.org/pdf/2510.01152)" by [Mustafa Omer Gul](https://momergul.github.io/), [Claire Cardie](https://www.cs.cornell.edu/home/cardie/) and [Tanya Goyal](https://tagoyal.github.io/) (arXiv preprint).

### About
LLMs cannot reliably recognize their parametric knowledge boundaries and often hallucinate answers to outside-of-boundary questions. In contrast, humans recognize their limitations and can either seek external help for such questions or abstain. In this paper, we introduce MASH (Modeling Abstention via Selective Help-seeking), a training framework that readily extracts abstentions from LLMs. Our key idea is that any external help-seeking by an LLM, i.e. search tool use, can serve as a proxy for abstention if the external help (search) is appropriately penalized while simultaneously rewarding answer accuracy. MASH operationalizes this idea using reinforcement learning with a pay-per-search reward.

We run experiments on three knowledge-intensive QA datasets. Our results show that MASH substantially improves upon the selective help-seeking performance of prior efficient search approaches; on multi-hop datasets, MASH improves answer accuracy by 7.6%. Furthermore, MASH demonstrates strong off-the-shelf abstention – it can distinguish between unanswerable/answerable questions and selectively generate responses for answerable questions – showcasing behavior analogous to specialized abstention approaches. We emphasize that contrary to prior abstention methods, MASH does not require pre-determining knowledge boundaries to construct training data. Instead, MASH’s abstentions are a by-product of training for the auxiliary selective help-seeking task. Overall, we show that MASH training effectively aligns search tool use with parametric knowledge, which can be successfully leveraged for making abstention decisions

Setup
-----

### Setting up the environment
1. To set up the environment for training/evaluation, execute the following code block, which will create a conda environment, install dependencies and alter the directory structure:
```
conda create -n mash python=3.10
conda activate mash
bash setup/create_mash_environment.sh
``` 
N.B.: If you receive flash attention errors when doing RL training, downgrading to version 2.7.3 can help.

2. To set up the environment for the retriever, we follow the procedure outlined in the [Search-R1 repository](https://github.com/PeterGriffinJin/Search-R1):
```
conda create -n retriever python=3.10
conda activate retriever

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install uvicorn fastapi
```

### Downloading the data

1. First download the Wikipedia dump needed for the retriever by running:
```
bash setup/download_retrieval_data.sh
``` 

2. Then run the following to download data particular to this project:
```
bash setup/download_mash_data.sh
``` 
This script will place all data under a `data` folder, which will contain .parquet dataset files used in RL training under `shortform_qa`, the question-answer pairs used to estimate question answerability under `model_accuracy_estimates`, and the SFT data used to warm-start models for MASH training under the `naive_sft` folder. The script will also create an `experiments` folder containing the test set predictions of the models trained and evaluated in the paper.

RL Training
-----------
1. Most RL training in the repository assumes a retriever running in the background. First initialize this by executing:
```
conda activate retriever

# To initialize on CPU
python -m verl_training.search.search_r1_server --listen_port=<retriever_port> &

# To initialize on GPU
python -m verl_training.search.search_r1_server --listen_port=<retriever_port> --faiss_gpu &
```

2. Afterwards, you can execute the following scripts to run the various kinds of RL training outlined in the paper (with argument options detailed inside the scripts themselves):
```
conda activate mash

# To run training with a warm-started model
bash example_scripts/mash_training/mash_training.sh <dataset_name> <search_penalty> <retriever_port>

# To run training without any warm-start
bash example_scripts/mash_training/mash_training.sh <dataset_name> <search_penalty> <retriever_port>

# To run Search-R1 training
bash example_scripts/mash_training/search_r1_training.sh <dataset_name> <retriever_port>

# To run R1 training
bash example_scripts/mash_training/r1_training.sh <dataset_name>
```

3. You can then evaluate any trained model by running
```
bash example_scripts/mash_training/evaluation.sh <dataset_name> <search_penalty> <test_dataset_name> <API_KEY> <model_loading_metric> <retriever_port>
```

Alternatively, if you wish to do evaluation with one of our trained models, run:
```
bash example_scripts/mash_training/paper_model_evaluation.sh <model_name> <test_dataset_name> <API_KEY> <retriever_port>
``` 
All trained models can be found in Huggingface, with the prefix `momergul/mash_`.

Warm-Start Training
-------------------
If you wish to execute the warm-start training process from scratch, run the following:

```
bash example_scripts/warm_start/generte_warm_start_data.sh <dataset_name>
bash example_scripts/warm_start/train_warm_start_model.sh <dataset_name>
```
Variants of the scripts with the oracle helper can also be found under `example_scripts/warm_start`.

Abstention Training and Inference
---------------------------------
1. If you wish to run abstention training from scratch, run the following:

```
# For AFH-Absolute
bash example_scripts/abstention/train_afh.sh <dataset_name> absolute

# For AFH-Multisample
bash example_scripts/abstention/train_afh.sh <dataset_name> multisample

# For DPO
bash example_scripts/abstention/train_dpo.sh <dataset_name>
```
Note that you will need to update your TRL version to `0.23.0` to run training for DPO, as the version on the codebase does not support DPO with an added SFT loss.

2. To perform inference/evaluation with abstention models, run:

```
# For AFH variants and DPO
bash example_scripts/abstention/trained_model_inference_and_eval.sh <experiment_name> <test_dataset_name> <checkpoint_name> <api_key>

# For few-shot prompting
bash example_scripts/abstention/few_shot_model_inference_and_eval.sh <few_shot_dataset_name> <test_dataset_name> <api_key>
```

Result Reporting
----------------
The `experiments` folder that `bash setup/download_mash_data.sh` constructs contains test set predictions for the models evaluated in the paper. To get the results, simply run:

```
# For RL model accuracy and abstention performance, as well as search count distribution:
bash example_scripts/result_reporting/mash_model_tool_use.sh <experiment_name> <load_metric>
bash example_scripts/result_reporting/mash_model_abstention.sh <experiment_name> <load_metric>
bash example_scripts/result_reporting/mash_model_search_distribution.sh <experiment_name> <load_metric>

# For abstention model performance:
bash example_scripts/result_reporting/trained_abstention_model_results.sh <experiment_name> <test_dataset_name>
bash example_scripts/result_reporting/few_shot_abstention_model_results.sh <experiment_name> <test_dataset_name>
```
