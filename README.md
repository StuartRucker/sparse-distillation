# Sparse Distillation

### Resources

Main Paper [https://arxiv.org/pdf/2110.08536.pdf](https://arxiv.org/pdf/2110.08536.pdf)

Dan code: [https://github.com/miyyer/dan](https://github.com/miyyer/dan)

Pytorch style guide: [https://github.com/IgorSusmelj/pytorch-styleguide](https://github.com/IgorSusmelj/pytorch-styleguide)

### Beta Run

Fine tuned a DAN with less layers just the training reviews, using a premade sentiment classier. in `v1.py`

- [x]  Run Count Vectorizer on the Training
- [x]  Use preexisting labels
- [x]  Run DAN on the training
- [x]  Evaluate the Dan on the test

### Preliminary Scaling

Scale up to using amazon reviews, Same parameter counts as paper, Still use roberta-large

- [ ]  Clean up code
- [ ]  Separate data preprocessing tasks
    - [ ]  CountTokenizer
    - [ ]  Sentiment
- [ ]  Model Checkpoints
- [ ]  MIT Satori

 

### Questions

- Variable Length spans
- Dropout after embeddings

### Pytorch Notes

- *BackgroundGenerator*
- manual seed
- parallelize across gpus dataparralel
- sparse gradient
- `torch.backends.cudnn.benchmark = True`


cat sentiment_reviews/* | shuf | split -l 11000 - sentiment_reviews_shuffled/reviews-

