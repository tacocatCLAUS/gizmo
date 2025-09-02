from yacana import OpenAiAgent, GenericMessage, Task

# Note the endpoint parameter is set to the VLLM server address
vllm_agent = OpenAiAgent("AI assistant", "meta-llama/llama-4-maverick-17b-128e-instruct", system_prompt="You are a helpful AI assistant", endpoint="https://ai.hackclub.com", api_token="leave blank")

# Use the agent to solve a task
message: GenericMessage = Task("What is the capital of France?", vllm_agent).solve()
print(message.content)