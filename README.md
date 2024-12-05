
```markdown
# Custom Llama LLM Integration

This project integrates a custom LLM (Llama-3.2-1B-Instruct) for natural language processing tasks. It uses the Llama Index library along with NVIDIA's embedding model to handle queries and responses efficiently.

## Features
- Custom Llama LLM with a context window of 3900 and a maximum output of 256 tokens.
- Integrates with NVIDIA's embedding model for enhanced performance.
- Allows query processing and response generation from a server running at a specified URL.


```

### 2. Install dependencies:
You need to install the following Python packages:
- llama-index
- requests
- NVIDIA Embeddings (optional, if using the NVIDIA model)

To install dependencies, run:
```bash
pip install llama-index requests
```

## Code

```python
from typing import Optional, List, Mapping, Any
from llama_index.core import SimpleDirectoryReader, SummaryIndex
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import Settings
import requests

class CustomLlamaLLM(CustomLLM):
    context_window: int = 3900
    num_output: int = 256
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    server_url: str = "http://localhost:port/v1/chat/completions"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    def _make_request(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        try:
            response = requests.post(self.server_url, json=payload, headers=headers)
            if response.status_code == 200:
                # Assuming the response JSON has a structure like {"choices": [{"message": {"content": "..."}}]}
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code}"
        except requests.RequestException as e:
            return f"Request failed: {str(e)}"

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response_text = self._make_request(prompt)
        return CompletionResponse(text=response_text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        response_text = self._make_request(prompt)
        response = ""
        for token in response_text:
            response += token
            yield CompletionResponse(text=response, delta=token)


Settings.llm = CustomLlamaLLM()
Embeddings = NVIDIAEmbedding(base_url="http://localhost:port/v1", model="nvidia/nv-embedqa-e5-v5")
Settings.embed_model = Embeddings

documents = SimpleDirectoryReader("data").load_data()
index = SummaryIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("what is github")
print(response)
```

## Usage

1. Place your data inside the `data` folder.
2. Run the script to start querying the Llama LLM model:

   ```bash
   python main.py
   ```

3. The code will query the Llama model for the input and return the response. Example:

   ```python
   response = query_engine.query("What is GitHub?")
   print(response)
   ```

## Customization

- Modify the `server_url` and `model_name` to use a different LLM model or server.
- Update the `NVIDIAEmbedding` parameters if you wish to use a different embedding model.

```

### Key Points:
- The `README.md` includes all the necessary instructions and the entire code block.
