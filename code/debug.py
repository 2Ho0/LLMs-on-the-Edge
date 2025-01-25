from transformers import pipeline
import torch

print("CUDA Available:", torch.cuda.is_available())  # True여야 GPU 사용 가능

qa_pipeline = pipeline(
"question-answering",
model="deepset/roberta-base-squad2",
tokenizer="deepset/roberta-base-squad2",
device=0
)

print("Pipeline Device:", qa_pipeline.device)  # 할당된 디바이스 출력
