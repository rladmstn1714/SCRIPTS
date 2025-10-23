"""
Model wrapper classes for various LLM providers

This module provides unified interfaces for different LLM models including:
- OpenAI (GPT models)
- Google (Gemini models)  
- Anthropic (Claude models)
- HuggingFace models (via vLLM)
- OpenRouter (unified API)

All API keys must be set as environment variables.
"""
import os
import json
import requests
from dotenv import load_dotenv

# Lazy imports - only import when needed to avoid dependency errors
try:
    from langchain.schema import SystemMessage, HumanMessage, AIMessage
except ImportError:
    # Fallback simple message classes
    class AIMessage:
        def __init__(self, content):
            self.content = content
    
    class HumanMessage:
        def __init__(self, content):
            self.content = content
    
    class SystemMessage:
        def __init__(self, content):
            self.content = content

# Load environment variables
load_dotenv()

# API key validation helper
def get_api_key(key_name, service_name):
    """Get API key from environment variables or raise error"""
    key = os.getenv(key_name)
    if not key:
        raise ValueError(
            f"\n{'='*60}\n"
            f"ERROR: {service_name} API key not found!\n"
            f"Please set the '{key_name}' environment variable.\n"
            f"{'='*60}\n"
        )
    return key

# Model configurations with GPU requirements
model_configs = {
    # Korean models
    "midm-2.0-11.5b": {
        "path": "K-intelligence/Midm-2.0-Base-Instruct",
        "gpu_memory": "24GB",
        "tensor_parallel_size": 2,
        "trust_remote_code": True
    },
    "exaone-4.0-32b": {
        "path": "LGAI-EXAONE/EXAONE-4.0.1-32B",
        "gpu_memory": "48GB",
        "tensor_parallel_size": 1,
        "trust_remote_code": True
    },
    "exaone-4.0-1.3b": {
        "path": "LGAI-EXAONE/EXAONE-4.0.1-1.3B",
        "gpu_memory": "8GB",
        "tensor_parallel_size": 1,
        "trust_remote_code": True
    },
    "kanana-1.5-8b": {
        "path": "kakaocorp/kanana-1.5-8b-instruct-2505",
        "gpu_memory": "16GB",
        "tensor_parallel_size": 1,
        "trust_remote_code": True
    },
    "kanana-1.5-15.7b": {
        "path": "kakaocorp/kanana-1.5-15.7b-a3b-instruct",
        "gpu_memory": "32GB",
        "tensor_parallel_size": 2,
        "trust_remote_code": True
    },
    "qwen3-14b": {
        "path": "Qwen/Qwen3-14B",
        "gpu_memory": "32GB",
        "tensor_parallel_size": 2,
        "trust_remote_code": True
    },
    "qwen3-32b": {
        "path": "Qwen/Qwen3-32B",
        "gpu_memory": "32GB",
        "tensor_parallel_size": 4,
        "trust_remote_code": True
    },
    "ax-4.0-light": {
        "path": "skt/A.X-4.0-Light",
        "gpu_memory": "16GB",
        "tensor_parallel_size": 1,
        "trust_remote_code": True
    },
    
    # API-based models
    'gpt-4.1-3': "gpt-4.1-2025-04-14",
    'gpt-5': "gpt-5-2025-08-07",
    "gpt-o3": "o3-2025-04-16",
    "gemini-1.5-pro": "gemini-1.5-pro",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.5-flash-thinking": "google/gemini-2.5-flash-thinking",
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4.1-mini": "openai/gpt-4.1-mini",
    "phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
    "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "hcx-003": "HCX-003",
    "hc-seed-15b": "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
    "exaone-32b": "LGAI-EXAONE/EXAONE-3.5-32B-Instruct",
    "exaone-8b": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "llama-3.1-8b-chat": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistral-7B-Instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.1",
    "deepseek-math-7b-instruct": "deepseek-ai/deepseek-math-7b-instruct",
    "deepseek-coder-7b-instruct-v1.5": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "qwen2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    'llama-3.1-8b-instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    "gemma-9b": "google/gemma-2-9b-it",
    "gemma-27b": "google/gemma-2-27b-it",
    'google/gemini-2.0-flash-001': 'google/gemini-2.0-flash-001',
    "olmoe": "allenai/OLMoE-1B-7B-0924",
    "claude-3.7-sonnet": "anthropic/claude-3.7-sonnet",
    "qwen3-8b": 'Qwen/Qwen3-8B',
}


class TestModel:
    """Test model for debugging"""
    def invoke(self, prompt):
        return AIMessage(content="test")


class HuggModel:
    """
    HuggingFace model wrapper using vLLM for efficient inference
    
    Supports local models with multi-GPU parallelism
    """
    def __init__(self, name, max_tokens=2048, temperature=0.7, top_p=0.9, 
                 tensor_parallel_size=None, thinking_budget=0, gpu_memory_utilization=0.9):
        self.name = name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tensor_parallel_size = tensor_parallel_size
        self.thinking_budget = thinking_budget
        self.gpu_memory_utilization = gpu_memory_utilization
        
        # Check if this is a configured model
        if name in model_configs and isinstance(model_configs[name], dict):
            config = model_configs[name]
            model_path = config["path"]
            tensor_parallel_size = config.get("tensor_parallel_size", tensor_parallel_size)
            trust_remote_code = config.get("trust_remote_code", True)
            
            print(f"Loading configured model: {name}")
            print(f"  Path: {model_path}")
            print(f"  Tensor parallel size: {tensor_parallel_size}")
            print(f"  GPU memory: {config.get('gpu_memory', 'Unknown')}")
            
            # Special handling for exaone-4.0-32b model
            if "exaone-4.0-32b" in name:
                print("Using transformers directly for exaone-4.0-32b")
                try:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    import torch
                except ImportError:
                    raise ImportError("transformers and torch packages are required. Install with: pip install transformers torch")
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.use_transformers = True
                print(f"✓ Model loaded successfully: {name}")
                return
            else:
                # Initialize with multi-GPU support
                try:
                    from vllm import LLM, SamplingParams
                except ImportError:
                    raise ImportError("vllm package is required. Install with: pip install vllm")
                
                self.model = LLM(
                    model=model_path,
                    tensor_parallel_size=tensor_parallel_size,
                    trust_remote_code=trust_remote_code,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=8192
                )
        else:
            # Legacy support for direct model names
            if 'mnt' in self.name:
                self.model = LLM(self.name, trust_remote_code=True)  
            else:
                self.model = LLM(self.name)
        
        print(f"✓ Model loaded successfully: {name}")
    
    def invoke(self, prompt):
        """Generate response from the model"""
        if hasattr(self, 'use_transformers') and self.use_transformers:
            # Use transformers directly
            try:
                import torch
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                return AIMessage(content=response.strip())
            except Exception as e:
                print(f"Error in HuggModel.invoke (transformers): {e}")
                return AIMessage(content="error")
        else:
            # Use vLLM
            try:
                from vllm import LLM, SamplingParams
            except ImportError:
                raise ImportError("vllm package is required for local model inference. Install with: pip install vllm")
            
            sampling_params = SamplingParams(
                temperature=self.temperature, 
                top_p=self.top_p, 
                max_tokens=self.max_tokens
            )
            
            if "qwen3-14b" in self.name and self.thinking_budget:
                print(f"Thinking enabled with budget: {self.thinking_budget}")
                sampling_params = SamplingParams(
                    temperature=self.temperature, 
                    top_p=self.top_p, 
                    max_tokens=self.thinking_budget
                )
                enable_thinking = True
            else:
                enable_thinking = False
            
            try:
                if "qwen3-14b" in self.name and enable_thinking:
                    text = self.tokenizer.apply_chat_template(
                        prompt,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                    outputs = self.model.generate([text], sampling_params)
                else:
                    outputs = self.model.generate([prompt], sampling_params)
                
                message = outputs[0].outputs[0].text.strip()
                return AIMessage(content=message)
            except Exception as e:
                print(f"Error in HuggModel.invoke: {e}")
                return AIMessage(content="error")


class GeminiModel:
    """
    Google Gemini model wrapper
    
    Requires: GEMINI_API_KEY environment variable
    """
    def __init__(self, name, temperature=0.7, max_tokens=512):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai package is required for Gemini models. Install with: pip install google-generativeai")
        
        self.name = name
        api_key = get_api_key('GEMINI_API_KEY', 'Google Gemini')
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(self.name)
        self.generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=0.95,
            top_k=40
        )

    def invoke(self, prompt):
        """Generate response from Gemini"""
        try:
            outputs = self.model.generate_content(prompt)
            message = outputs.candidates[0].content.parts[0].text
            return AIMessage(content=message)
        except Exception as e:
            print(f"Error in GeminiModel.invoke: {e}")
            return AIMessage(content="error")


class OpenrouterModel:
    """
    OpenRouter model wrapper for unified API access
    
    Requires: OPENROUTER_API_KEY environment variable
    """
    def __init__(self, name, temperature=0.7, max_tokens=512, api_key=None,
                 thinking_budget=None, reasoning_enabled=False, 
                 reasoning_budget=None, reasoning_effort=None):
        self.name = name
        self.temp = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or get_api_key('OPENROUTER_API_KEY', 'OpenRouter')
        self.thinking_budget = thinking_budget
        self.reasoning_enabled = reasoning_enabled
        self.reasoning_budget = reasoning_budget
        self.reasoning_effort = reasoning_effort

    def invoke(self, prompt):
        """Generate response via OpenRouter API"""
        try:
            # Build request data
            data = {
                "model": self.name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": self.temp,
                "max_tokens": self.max_tokens
            }
            
            # Add thinking_budget if provided
            if self.thinking_budget is not None:
                data["thinking_budget"] = self.thinking_budget
                print(f"Using thinking_budget: {self.thinking_budget}")
            
            # Add reasoning if enabled
            if self.reasoning_enabled:
                reasoning_config = {"enabled": True}
                if self.reasoning_effort is not None:
                    reasoning_config["effort"] = self.reasoning_effort
                elif self.reasoning_budget is not None:
                    reasoning_config["max_tokens"] = self.reasoning_budget
                reasoning_config["exclude"] = True
                data["reasoning"] = reasoning_config
                print(f"Using reasoning: {reasoning_config}")

            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                },
                data=json.dumps(data)
            )

            response_json = response.json()
            message = response_json['choices'][0]['message']['content']
            
            # Check for reasoning tokens in response
            if 'reasoning' in response_json['choices'][0]['message']:
                reasoning = response_json['choices'][0]['message']['reasoning']
                print(f"Reasoning tokens: {reasoning}")
            
            return AIMessage(content=message)
        except Exception as e:
            print(f"Error in OpenrouterModel.invoke: {e}")
            return AIMessage(content="error")


class ChatModel:
    """
    Main model factory class
    
    Creates appropriate model instances based on model name
    """
    
    @staticmethod
    def create_model(name, temp=0.7, max_tokens=2048, tensor_parallel_size=None,
                    reasoning_enabled=False, reasoning_budget=None, 
                    reasoning_effort=None, thinking_budget=None):
        """
        Create a model instance
        
        Args:
            name: Model name (from model_configs)
            temp: Temperature for generation
            max_tokens: Maximum tokens to generate
            tensor_parallel_size: Number of GPUs for parallel inference
            reasoning_enabled: Enable reasoning mode (for supported models)
            reasoning_budget: Token budget for reasoning
            reasoning_effort: Reasoning effort level
            thinking_budget: Thinking token budget
            
        Returns:
            Model instance with invoke() method
        """
        model_name = name.lower()
        
        if model_name not in model_configs:
            raise ValueError(
                f"\n{'='*60}\n"
                f"ERROR: Model '{model_name}' is not supported!\n"
                f"Available models: {list(model_configs.keys())}\n"
                f"{'='*60}\n"
            )
        
        # Test model
        if 'test' in model_name:
            return TestModel()
        
        # OpenAI GPT models
        elif "gpt" in model_name:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise ImportError("langchain-openai package is required. Install with: pip install langchain-openai")
            
            openai_key = get_api_key('OPENAI_API_KEY', 'OpenAI')
            if "o3" in model_name:
                max_tokens = 2048
            
            return ChatOpenAI(
                model=model_configs[model_name],
                temperature=temp,
                max_tokens=max_tokens,
                timeout=None,
                max_retries=2,
                api_key=openai_key
            )
        
        # Claude models via OpenRouter
        elif "claude" in model_name:
            return OpenrouterModel(
                name=model_configs[model_name],
                temperature=temp,
                max_tokens=max_tokens
            )
        
        # Gemini models via OpenRouter (for reasoning/thinking support)
        elif "gemini" in model_name:
            return OpenrouterModel(
                name=model_configs[model_name],
                temperature=temp,
                max_tokens=max_tokens,
                reasoning_enabled=reasoning_enabled,
                reasoning_budget=reasoning_budget,
                reasoning_effort=reasoning_effort,
                thinking_budget=thinking_budget
            )
        
        # HyperCLOVA X
        elif "hcx" in model_name:
            try:
                from langchain_community.chat_models import ChatClovaX
            except ImportError:
                raise ImportError("langchain-community package is required. Install with: pip install langchain-community")
            
            ncp_key = get_api_key('NCP_CLOVASTUDIO_API_KEY', 'NAVER HyperCLOVA X')
            return ChatClovaX(
                model="HCX-003",
                ncp_clovastudio_api_key=ncp_key,
                max_tokens=max_tokens,
                temperature=temp
            )
        
        # Korean and other HuggingFace models
        elif any(keyword in model_name for keyword in ["midm", "kanana", "ax-4.0", "qwen", "exaone", "seed", "llama", "mixtral", "mistral", "deepseek", "gemma", "olmoe"]):
            if isinstance(model_configs[model_name], dict):
                model_path = model_configs[model_name]['path']
            else:
                model_path = model_configs[model_name]
            
            return HuggModel(
                model_path,
                max_tokens=max_tokens,
                temperature=temp,
                tensor_parallel_size=tensor_parallel_size,
                thinking_budget=thinking_budget
            )
        
        else:
            raise ValueError(
                f"\n{'='*60}\n"
                f"ERROR: Don't know how to create model '{model_name}'\n"
                f"{'='*60}\n"
            )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="test")
    args = parser.parse_args()
    
    try:
        model = ChatModel.create_model(args.model)
        response = model.invoke("Hello, how are you?")
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Error: {e}")


                max_tokens=max_tokens,
                temperature=temp,
                tensor_parallel_size=tensor_parallel_size,
                thinking_budget=thinking_budget
            )
        
        else:
            raise ValueError(
                f"\n{'='*60}\n"
                f"ERROR: Don't know how to create model '{model_name}'\n"
                f"{'='*60}\n"
            )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="test")
    args = parser.parse_args()
    
    try:
        model = ChatModel.create_model(args.model)
        response = model.invoke("Hello, how are you?")
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Error: {e}")

