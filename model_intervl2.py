from vllm import LLM



# For pooling tasks
llm = LLM(model='OpenGVLab/InternVL2_5-1B', task='embed', dtype='float16')
embedding = llm.encode("Describe the image.")
print(embedding)