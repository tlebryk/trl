.PHONY: help build shell pipeline eval clean

# Docker image name
IMAGE_NAME := dpo
CONTAINER_NAME := dpo-container

# Directories that need to be mounted
PROJECT_ROOT := $(shell pwd)
COMPLETIONS_DIR := $(PROJECT_ROOT)/completions
EVAL_RESULTS_DIR := $(PROJECT_ROOT)/eval_results
CPP_PIPELINE_DIR := $(PROJECT_ROOT)/cpp_pipeline
TRAINING_DIR := $(PROJECT_ROOT)/training

# Ensure output directories exist
$(EVAL_RESULTS_DIR):
	mkdir -p $(EVAL_RESULTS_DIR)

$(COMPLETIONS_DIR):
	mkdir -p $(COMPLETIONS_DIR)

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Build the Docker image
	docker build -t $(IMAGE_NAME) .

shell: build $(EVAL_RESULTS_DIR) $(COMPLETIONS_DIR) ## Open an interactive bash shell in the container
	docker run --rm -it \
		--name $(CONTAINER_NAME) \
		-v $(PROJECT_ROOT):/app \
		-v $(COMPLETIONS_DIR):/completions:ro \
		-v $(EVAL_RESULTS_DIR):/eval_results \
		-w /app \
		$(IMAGE_NAME) \
		bash

run: shell ## Alias for shell command

pipeline: build ## Run the complete C++ pipeline (create, compile, run, compute rewards)
	docker run --rm \
		-v $(PROJECT_ROOT):/app \
		-w /app \
		$(IMAGE_NAME) \
		uv run python -m cpp_pipeline.run_pipeline

pipeline-stage1: build ## Stage 1: Create examples
	docker run --rm \
		-v $(PROJECT_ROOT):/app \
		-w /app \
		$(IMAGE_NAME) \
		uv run python -m cpp_pipeline.create_examples

pipeline-stage2: build ## Stage 2: Compile examples
	docker run --rm \
		-v $(PROJECT_ROOT):/app \
		-w /app \
		$(IMAGE_NAME) \
		uv run python -m cpp_pipeline.compile_examples

pipeline-stage3: build ## Stage 3: Run examples (runtime checks)
	docker run --rm \
		-v $(PROJECT_ROOT):/app \
		-w /app \
		$(IMAGE_NAME) \
		uv run python -m cpp_pipeline.run_examples

pipeline-stage4: build ## Stage 4: Compute token-level rewards
	docker run --rm \
		-v $(PROJECT_ROOT):/app \
		-w /app \
		$(IMAGE_NAME) \
		uv run python -m cpp_pipeline.compute_rewards

pipeline-stage5: build ## Stage 5: Prepare dataset
	docker run --rm \
		-v $(PROJECT_ROOT):/app \
		-w /app \
		$(IMAGE_NAME) \
		uv run python -m cpp_pipeline.prepare_dataset

eval: build ## Evaluate completions (FILE=path/to/completions.json, OUTPUT=optional output path)
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make eval FILE=eval_completions_test.json"; \
		echo "  Or: make eval FILE=completions/completions_dpo-eval.json OUTPUT=eval_results/eval_dpo.json"; \
		echo ""; \
		echo "Results will be saved next to the input file if OUTPUT is not specified."; \
		exit 1; \
	fi
	@if [ -n "$(OUTPUT)" ]; then \
		docker run --rm \
			-v $(PROJECT_ROOT):/app \
			-w /app \
			$(IMAGE_NAME) \
			uv run python training/eval_completions.py \
				$(FILE) \
				$(OUTPUT); \
	else \
		docker run --rm \
			-v $(PROJECT_ROOT):/app \
			-w /app \
			$(IMAGE_NAME) \
			uv run python training/eval_completions.py \
				$(FILE); \
	fi
	@echo ""; \
	echo "✅ Evaluation complete!"; \
	if [ -n "$(OUTPUT)" ]; then \
		echo "📄 Results: $(OUTPUT)"; \
		echo "📁 Artifacts: $$(dirname $(OUTPUT))/$$(basename $(OUTPUT) .json)_artifacts/"; \
	else \
		INPUT_DIR=$$(dirname $(FILE)); \
		INPUT_BASE=$$(basename $(FILE) .json); \
		echo "📄 Results: $$INPUT_DIR/eval_$$INPUT_BASE.json"; \
		echo "📁 Artifacts: $$INPUT_DIR/eval_$$INPUT_BASE_artifacts/"; \
	fi

# Modal training commands (run locally, execute on Modal)
train-dpo: ## Train DPO model on Modal (requires experiment name)
	@if [ -z "$(EXP)" ]; then \
		echo "Usage: make train-dpo EXP=experiment-name"; \
		exit 1; \
	fi
	uv run modal run training/modal_train_dpo.py --experiment-name $(EXP)

train-dpo-baseline: ## Train vanilla DPO baseline on Modal
	@if [ -z "$(EXP)" ]; then \
		echo "Usage: make train-dpo-baseline EXP=experiment-name"; \
		exit 1; \
	fi
	uv run modal run training/modal_train_dpo.py --experiment-name $(EXP) --no-use-token-level-rewards

train-ppo: ## Train PPO model on Modal
	@if [ -z "$(EXP)" ]; then \
		echo "Usage: make train-ppo EXP=experiment-name"; \
		exit 1; \
	fi
	uv run modal run training/modal_train_ppo.py --experiment-name $(EXP)

train-ppo-baseline: ## Train vanilla PPO baseline on Modal
	@if [ -z "$(EXP)" ]; then \
		echo "Usage: make train-ppo-baseline EXP=experiment-name"; \
		exit 1; \
	fi
	uv run modal run training/modal_train_ppo.py --experiment-name $(EXP) --no-use-token-level-rewards

train-grpo: ## Train GRPO model on Modal
	@if [ -z "$(EXP)" ]; then \
		echo "Usage: make train-grpo EXP=experiment-name"; \
		exit 1; \
	fi
	uv run modal run training/modal_train_grpo.py --experiment-name $(EXP)

train-grpo-baseline: ## Train vanilla GRPO baseline on Modal
	@if [ -z "$(EXP)" ]; then \
		echo "Usage: make train-grpo-baseline EXP=experiment-name"; \
		exit 1; \
	fi
	uv run modal run training/modal_train_grpo.py --experiment-name $(EXP) --no-use-token-level-rewards

# Modal inference commands
inference: ## Generate completions on Modal
	@if [ -z "$(ADAPTER)" ] || [ -z "$(RUN)" ]; then \
		echo "Usage: make inference ADAPTER=path/to/adapter RUN=run-name"; \
		echo "  Example: make inference ADAPTER=dpo-token-rewards-v1/final_model RUN=dpo-eval"; \
		exit 1; \
	fi
	uv run modal run training/modal_inference.py \
		--adapter-path $(ADAPTER) \
		--run-name $(RUN)

inference-base: ## Generate completions with base model (no fine-tuning)
	@if [ -z "$(RUN)" ]; then \
		echo "Usage: make inference-base RUN=run-name"; \
		exit 1; \
	fi
	uv run modal run training/modal_inference.py --run-name $(RUN)

# Modal evaluation commands
eval-modal: ## Evaluate completions on Modal
	@if [ -z "$(COMPLETIONS)" ]; then \
		echo "Usage: make eval-modal COMPLETIONS=/data/inference_results/completions_dpo-eval.json"; \
		exit 1; \
	fi
	uv run modal run training/modal_eval_completions.py \
		--completions-path $(COMPLETIONS)

# Modal download commands
download-model: ## Download trained model from Modal
	@if [ -z "$(EXP)" ]; then \
		echo "Usage: make download-model EXP=experiment-name"; \
		exit 1; \
	fi
	modal volume get --force dpo-training-vol /experiments/$(EXP) ./results/$(EXP)

download-completions: ## Download completions from Modal
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make download-completions FILE=completions_dpo-eval.json"; \
		exit 1; \
	fi
	modal volume get dpo-training-vol /data/inference_results/$(FILE) ./completions/

download-eval: ## Download evaluation results from Modal
	@if [ -z "$(RUN)" ]; then \
		echo "Usage: make download-eval RUN=run-name"; \
		exit 1; \
	fi
	modal volume get dpo-training-vol /data/inference_results/eval_completions_$(RUN).json ./
	modal volume get dpo-training-vol /data/inference_results/eval_completions_$(RUN)_artifacts ./

list-experiments: ## List all experiments on Modal volume
	modal volume ls dpo-training-vol /experiments

clean: ## Clean up generated files (examples, compiled, rewards)
	rm -rf cpp_pipeline/examples/*/chosen.cpp
	rm -rf cpp_pipeline/examples/*/rejected.cpp
	rm -rf cpp_pipeline/examples/*/metadata.json
	rm -rf cpp_pipeline/compiled/*
	rm -rf cpp_pipeline/rewards/*
	@echo "Cleaned up generated pipeline files"

clean-all: clean ## Clean everything including eval results and completions
	rm -rf $(EVAL_RESULTS_DIR)/*
	rm -rf $(COMPLETIONS_DIR)/*
	@echo "Cleaned up all generated files"
