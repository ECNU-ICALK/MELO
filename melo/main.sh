+alg=lora +experiment=qa +model=t5small
+alg=lora +experiment=hallucination +model=gpt2xl
+alg=lora +experiment=scotus +model=scotus-bert


CUDA_VISIBLE_DEVICES=3 python few_shot_run.py +alg=lora +experiment=fnli +model=bert-base batch_size=10 val_batch_size=10 lora.cross_attend=True



CUDA_VISIBLE_DEVICES=3 python few_shot_run.py +alg=lora +experiment=qa +model=t5large batch_size=10 val_batch_size=10 data.zsre_impl=false data.zsre_yn=false data.hard_neg=false