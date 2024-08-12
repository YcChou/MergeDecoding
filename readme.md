# Merge Decoding
The Benefits in Shallow: Merge Decoding across Large Language Model Layers.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

Large language models (LLMs) have become foundational to
numerous natural language processing tasks; however, decoding coher-
ent and contextually relevant text remains a complex challenge. In open-
ended generation, maximizing probability is often not the appropriate
objective, as with sampling methods, the continuation tends to be inco-
herent and repetitive in various degrees. We propose Merge Decoding,
merging information in the shallow layer, such as sequential information,
with the final task-specific layer, thereby generating coherent and rich
text. MD works across three scales of the LLaMA family(7b, 13b, 30b),
achieving higher quality text in open-ended text generation (WikiText,
WikiNews, BookCorpus) and enhancing reasoning capabilities in down-
stream tasks (Gsm8k, StrategyQA).

## Installation

1. Create a Conda environment using the YAML file:

    ```bash
    conda env create -f environment.yml
    ```

## Usage

```bash
cd MergeDecoding
bash ./scripts/process_data.sh
bash ./scripts/wikitext/pipeline.sh
bash ./scripts/wikinews/pipeline.sh
bash ./scripts/bookcorpus/pipeline.sh
bash ./scripts/gsm8k/pipeline.sh
bash ./scripts/straqa/pipeline.sh
