\# Deep Learning OCR



This directory contains the Deep Learning stage of the handwriting OCR project.



The current implementation uses Microsoft's TrOCR handwritten text recognition model, fine-tuned on a custom handwritten dataset.



\---



\## Overview



The Deep Learning OCR workflow consists of:



1\. Zero-shot handwritten text recognition using pretrained TrOCR

2\. Fine-tuning TrOCR on a custom handwritten dataset

3\. Single-image handwritten word prediction

4\. Full-page handwritten text recognition through line segmentation

5\. Prescription image testing

6\. Experimental OCR approaches



The fine-tuned TrOCR model is stored locally because of its large size and is excluded from Git.



\---



\## Directory Structure



```text

deep\_learning/

├── 1\_zero\_shot\_baseline.py

├── 2\_train\_finetune.py

├── 3\_predict\_single\_word.py

├── 4\_full\_page\_showcase.py

├── 5\_prescription\_test.py

├── finetune\_data/

├── finetuned\_trocr/

├── experiments/

│   ├── 6\_ultimate\_ocr.py

│   ├── 7\_clean\_ai\_ocr.py

│   ├── 8\_projection\_ocr.py

│   └── 10\_word\_level\_ocr.py

├── word\_img.jpeg

├── word\_img2.jpg

└── cnn\_model.pth

