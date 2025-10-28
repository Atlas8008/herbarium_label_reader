#!/bin/bash

python extract_data.py \
    --multirun \
    image_list=data/handwritten.txt,data/printed.txt \
    preprocessors.grounding_dino.enabled=true,false \
    img_max_size=2048 \
    batch_size=1,5,20 \
    llm.model_name=gemini-2.5-flash,gemini-2.5-pro,llama-4-scout-17b-16e-instruct,llama-4-maverick-17b-128e-instruct,gpt-4.1-mini,gpt-4.1-nano,gpt-4o-mini \
    hydra.sweep.dir=outputs/experiments_output

