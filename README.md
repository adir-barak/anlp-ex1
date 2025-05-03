# Advanced NLP Exercise 1: Fine Tuning

This is the code base for ANLP HUJI course exercise 1, fine tuning 
pretrained models to perform sentiment analysis on the nyu-mll/glue MRPC 
dataset.

> Tested on Python 3.11.4 (Windows 11)
---

## Install
``` pip install -r requirements.txt ```

## Fine-Tune and Predict on Test Set
Run:

``` python ex1.py --max_train_samples <number of train samples> --max_eval_samples <number of validation samples> --max_predict_samples <number of prediction samples> --lr <learning rate> --num_train_epochs <number of training epochs> --batch_size <batch size> --do_train/--do_predict --model_path <path to prediction model>```

If you use --do_predict, a prediction.txt file will be generated, containing prediction results for all test samples.

---

## Model Comparison

To run the **qualitative analysis**, use the `compare_models.py` script:

* Compares best vs worst models on the validation set
* Extracts examples where the best was correct and the worst failed
* Outputs these to `comparison.txt` for manual inspection

---

## 📚 Notes

* Dataset used: MRPC (from GLUE benchmark)
* Model: `bert-base-uncased`
* Training tracked with Weights & Biases (wandb)
* Using `Trainer` from Hugging Face for training/prediction

---

For any issues, rerun with a few samples to debug:

```python ex1.py --do_train --max_train_samples 16 --max_eval_samples 16  --num_train_epochs 1 --lr 1e-5 --batch_size 16```
