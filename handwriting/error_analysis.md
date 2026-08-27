# Handwriting Model Error Analysis

## Current Baseline

The current HOG + SVM model was evaluated on 62 test images
containing digits, uppercase letters, and lowercase letters.

- Total test samples: 62
- Correct predictions: 37
- Incorrect predictions: 25
- Test accuracy: 59.68%

## Observed Confusions

### Digits

Some digits are confused with visually similar characters:

- 0 → C
- 1 → i
- 2 → p
- 4 → Q
- 5 → T
- 6 → 4
- 8 → e

### Uppercase Letters

Some uppercase characters are confused with similar characters:

- D → O
- J → 2
- O → o
- P → R

### Lowercase Letters

Several lowercase characters are confused with other lowercase
letters, digits, or uppercase letters:

- a → u
- c → e
- e → B
- f → r
- g → 9
- h → r
- i → j
- j → f
- k → B
- l → 1
- s → 3
- t → k
- u → 4
- v → Q

## Observation

The current baseline performs substantially better on uppercase
characters than on lowercase characters.

The observed errors suggest that visually similar characters
remain difficult for the current HOG + SVM approach.

## Next Step

Before modifying the model, the dataset and preprocessing pipeline
should be reviewed to determine whether the errors are caused by
image quality, character similarity, or limitations of the current
features and classifier.