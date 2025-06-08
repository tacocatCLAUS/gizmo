from tavily import TavilyClient
client = TavilyClient("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")
response = client.search(
    query="latest ipone news",
    include_answer="basic"
)
print(response)