
# Deep Learning OCR

This directory contains the Deep Learning stage of the handwriting OCR project.

The implementation uses Microsoft's TrOCR handwritten text recognition model, fine-tuned on a custom handwritten dataset.

## Overview

The Deep Learning OCR workflow includes:

1. Zero-shot handwritten text recognition using pretrained TrOCR
2. Fine-tuning TrOCR on a custom handwritten dataset
3. Single-image handwritten word prediction
4. Full-page handwritten text recognition through line segmentation
5. Prescription image testing
6. Experimental OCR approaches

The fine-tuned TrOCR model is stored locally because of its large size and is excluded from Git.

## Dataset

The fine-tuning dataset contains 100 handwritten images covering 20 distinct text labels, with 5 handwritten examples per label.

The labels include medicine names, units, and prescription-related phrases.

## Main Components

- 1_zero_shot_baseline.py - tests pretrained TrOCR without project-specific fine-tuning
- 2_train_finetune.py - fine-tunes TrOCR on the custom handwritten dataset
- 3_predict_single_word.py - performs single-image handwritten word prediction
- 4_full_page_showcase.py - performs line-based full-page recognition with dictionary correction
- 5_prescription_test.py - tests the model on a handwritten prescription image

## Experiments

The experiments directory contains additional OCR approaches explored during development:

- 6_ultimate_ocr.py
- 7_clean_ai_ocr.py
- 8_projection_ocr.py
- 10_word_level_ocr.py

## Testing

The fine-tuned model was tested on handwritten word and full-page examples.

The full-page showcase successfully recognized the tested clean handwritten page, while the prescription test showed that complex handwriting and overlapping content remain challenging.

The tested dictionary correction successfully corrected recognized terms such as Azithromycin, Dolo, Crocin, and Aspirin in the showcase example.

## Important Limitation

The Deep Learning stage demonstrates handwritten text recognition using a fine-tuned TrOCR model. It does not represent a formal benchmark accuracy evaluation because the available test examples are limited.

The large fine-tuned model files are intentionally excluded from Git.

## Status

Deep Learning OCR development and integration are complete.
