# OA4A - OpenAI for All
Extensible OpenAI API bridge supporting multiple LLM providers

## What does OA4A do?

Frequently run into cool and exciting projects that are built on top of OpenAI's API specs, but don't want to use OpenAI models? This project solves that by providing the bridge (or 'glue') between OpenAIs APIs and custom LLMs. 

## Supported providers and models

### Ollama

### Amazon Bedrock
- Claude3
- Llama2
- Titan

## Getting Started

```
git clone https://github.com/dgootman/oa4a.git
cd oa4a
pip install pipx
pipx install poetry
make
```

Running the server to invoke an Amazon Bedrock model

```
PROVIDER=bedrock AWS_PROFILE=<your-profile> make dev
```

## Example: Use with iTerm AI integration

Once the server is started with your desired provider:
1. Navigate to `iTerm > Settings > Advanced`.
2. Type in 'url' in the search bar, you should see `Experimental Features`. Specifically: 
   - `URL for AI API` 
3. Change this to `http://127.0.0.1:8000/v1/completions`
4. Now, navigate to `General`, under `AI` 
   - OpenAI API Key: Enter anything
   - Model: Change it to the model you want, for example, in the base of AWS Bedrock, this will be the model name. For 'Claude 3 Sonnet' it is `anthropic.claude-3-sonnet-20240229-v1:0` as mentioned [here](https://docs.anthropic.com/en/docs/models-overview#claude-3-a-new-generation-of-ai)
 5. You're good to go! Hit `⌘ + y` to engage AI, or `Edit > Engage Artificial Intelligence`.

## License

MIT, see [License](./LICENSE)