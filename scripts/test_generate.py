from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

prompts = ["用一句话介绍一下你自己。"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print("生成结果：", output.outputs[0].text)