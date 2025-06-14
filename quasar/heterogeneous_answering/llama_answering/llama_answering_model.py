import vllm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from loguru import logger

import quasar.heterogeneous_answering.llama_answering.dataset_llama_answering as dataset


class LLaMAModel:
    VLLM_TENSOR_PARALLEL_SIZE = 4
    VLLM_MAX_MODEL_LEN = 4096
    VLLM_GPU_MEMORY_UTILIZATION = 0.3

    DEFAULT_SAMPLING_PARAMS = {
        "seed": 7,
        "n": 1,
        "top_p": 0.8,
        "temperature": 0.0,
        "max_tokens": 4096,
        "skip_special_tokens": True
    }

    def __init__(self, config, train=True):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["answering_tokenizer_path"], cache_dir="checkpoints")
        self.tokenizer.pad_token = "<|finetune_right_pad_id|>"
        if train:
            logger.info("Loading base LLaMA model")
            self.model = AutoModelForCausalLM.from_pretrained(config["answering_model_base_path"], cache_dir="checkpoints", device_map="auto")
        else:
            logger.info("Loading fine-tuned LLaMA model")
            # self.model = AutoModelForCausalLM.from_pretrained(config["answering_model_inference_path"], cache_dir="checkpoints", device_map="auto")
            self.model = vllm.LLM(
                config["answering_model_inference_path"],
                tokenizer=config["answering_tokenizer_path"],
                distributed_executor_backend="ray",
                enforce_eager=True,
                tensor_parallel_size=self.VLLM_TENSOR_PARALLEL_SIZE,
                max_model_len=self.VLLM_MAX_MODEL_LEN,
                gpu_memory_utilization=self.VLLM_GPU_MEMORY_UTILIZATION
            )

    def batch_inference(self, input_texts, use_tqdm: bool=True):
        """
        Generates responses for a batch of dialogs.
        """
        return self._batch_inference(input_texts, sampling_params={}, use_tqdm=use_tqdm)

    def _batch_inference(self, llm_inputs, sampling_params, use_tqdm: bool=True):
        """
        Generates responses for a batch of formatted inputs.
        """
        # set sampling params
        s_params = self.DEFAULT_SAMPLING_PARAMS.copy()
        s_params.update(sampling_params)
        sampling_params = vllm.SamplingParams(**s_params)

        # run generation
        responses = self.model.generate(
            llm_inputs,
            sampling_params,
            use_tqdm=use_tqdm
        )

        # obtain list of outputs
        outputs = []
        for response in responses:
            output_texts = [output.text.strip() for output in response.outputs]
            # flatten single outputs
            if len(output_texts) == 1:
                output_texts = output_texts[0]
            outputs.append(output_texts)
        return outputs

    def inference(self, input_text): 
        return self.batch_inference([input_text])[0]
    
    def train(self, train_path, dev_path):
        logger.info(f"Cuda available: {torch.cuda.is_available()}")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")

        train_dataset = dataset.DatasetLLaMAAnswering(self.config, self.tokenizer, train_path)
        dev_dataset = dataset.DatasetLLaMAAnswering(self.config, self.tokenizer, dev_path)
       
        additional_keys = [("learning_rate", "answering_model_learning_rate")]
        additional_params = {
            param: self.config[load_name]
            for param, load_name in additional_keys
            if self.config.get(load_name)
        }

        # training args
        training_args = TrainingArguments(
            output_dir=self.config["answering_model_save_path"],
            warmup_ratio=self.config["answering_model_warmup_ratio"],
            num_train_epochs=self.config["answering_model_num_epochs"],
            per_device_train_batch_size=self.config["answering_model_train_batch_size"],
            remove_unused_columns=False,  # prevents from indexing errors
            save_strategy="epoch",  # epoch-wise eval
            logging_steps=20,
            eval_strategy="epoch",  # epoch-wise eval
            save_only_model=True,  # do not store optimizer state etc. to save space
            save_total_limit=1,  # only store best model
            report_to="none",  # avoid issues with distributed setup
            load_best_model_at_end="True",
            bf16=torch.cuda.is_bf16_supported(),  # mixed-precision training,
            **additional_params
        )

        # trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
        )

        logger.info("Starting training now...")
        trainer.train()
        logger.info("Done with training!")

        self.model.save_pretrained(self.config["answering_model_save_path"])

        