# Autoresearch Ant Detection

![AI for ants in action](assets/AI%20for%20ants%20in%20action.png)

A small adaptation of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for VS Code + Github Copilot, focused on detecting ants.


## What we do differently here

- Focus on using Github Copilot in VS Code. We include the settings for explicity locking down immutable files.
- Use python logging to generate the run.log file because the VS Code + Github Copilot combo does not allow auto approval of `uv run train.py > run.log 2>&1` file creation.
- Using Hugging Face Transformers/Trainer pipeline for object detection instead of an LLM. 
- We train on an ant detection dataset, because why not? 
- We have included journal.md and strategy.md files to aid in agent planning.
- Our harness (train.py, prepare.py, program.md) is structured slightly differently as we have been experimenting with generalizing them into general templates. 


## Quick start

Setup python environment
```bash
uv sync
```

Download the [dataset](https://www.kaggle.com/datasets/elizamoscovskaya/ant-2-keypoints-dataset-on-natural-substrate) from Kaggle and extract it to the expected layout below.

```text
.data/
	train/
		images/
		bboxes/
	test/
		images/
		bboxes/
```

Test data pipeline - this will export an annotated batch of images to .data/test_batches.
```bash
uv run prepare.py
```

Test training loop
```bash
uv run train.py
```

Start a new session with your Github Copilot agent in VS Code and ask. Work though initial setup and then during the first experiment auto-approve uv and git commands for the given session. This will allow the agent to run autonomously with minimal permissions until the agent execution time limit is reached.
```
Read program.md and let's kick off a new experiment.
```


## Project layout

- `prepare.py` defines dataset loading, evaluation, and task constraints
- `train.py` defines the model, augmentations, and training loop
- `results.tsv`, `journal.md`, `strategy.md`, and `program.md` capture experiment process and direction
- `.vscode/settings.json` locks down file permissions to maintain the integrity of each experiment run. Note that you will need to toggle read-only permissions to false to make any edits yourself.

## Attribution

This project is derived from and inspired by the original [karpathy/autoresearch](https://github.com/karpathy/autoresearch). If you want the broader framework and motivation, start there.

Dataset used in this project: [Ant 2 Keypoints Dataset on Natural Substrate](https://www.kaggle.com/datasets/elizamoscovskaya/ant-2-keypoints-dataset-on-natural-substrate)

Inspiration for journaling was derived from [yoyo-evolve](https://github.com/yologdev/yoyo-evolve)
