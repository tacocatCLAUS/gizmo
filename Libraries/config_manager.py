# config_manager.py
import json
from pathlib import Path
import os

if os.path.exists("model/config.json"):
    config_file = "model/config.json"  # Called from project root
else:
    config_file = "config.json"      # Called from RAG directory

class ConfigManager:
    def __init__(self, config_file="model/config.json"):
        self.config_file = Path(config_file)
        self.default_config = {
            "openai": False,
            "openai_model": "gpt-3.5-turbo",
            "openai_api_key": "",
            "hc": False,
            "hc_model": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "devmode": False,
            "db_clear": True,
            "use_mcp": True,
            "voice": False,
            "rag_model": "ollama",
            "mcp_config_path": "mcp.json"
        }
        self.load_config()
    
    def load_config(self):
        """Load configuration from JSON file, create if doesn't exist"""
        if not self.config_file.exists():
            self.save_config(self.default_config)
            return self.default_config
        
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Error loading config, using defaults")
            return self.default_config
    
    def save_config(self, config):
        """Save configuration to JSON file"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get(self, key, default=None):
        """Get a configuration value"""
        config = self.load_config()
        return config.get(key, default)
    
    def set(self, **kwargs):
        """Set configuration values"""
        config = self.load_config()
        config.update(kwargs)
        self.save_config(config)
        return config
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.save_config(self.default_config)
        return self.default_config

# Usage functions for easy access
def set_openai(enabled=True, model="gpt-3.5-turbo", api_key=None):
    """Enable OpenAI with optional model selection and API key"""
    config_manager = ConfigManager()
    config_update = {"openai": enabled, "openai_model": model, "hc": False}
    if api_key is not None:
        config_update["openai_api_key"] = api_key
    return config_manager.set(**config_update)

def set_hackclub(enabled=True, model="meta-llama/llama-4-maverick-17b-128e-instruct"):
    """Enable HackClub with optional model selection"""
    config_manager = ConfigManager()
    return config_manager.set(hc=enabled, hc_model=model, openai=False)

def set_ollama():
    """Switch to Ollama (disable OpenAI and HackClub)"""
    config_manager = ConfigManager()
    return config_manager.set(openai=False, hc=False)

def set_rag_model(model="ollama"):
    """Set the RAG model (ollama or openai)"""
    config_manager = ConfigManager()
    return config_manager.set(rag_model=model)

def set_mcp_config_path(path="mcp.json"):
    """Set the MCP configuration file path"""
    config_manager = ConfigManager()
    return config_manager.set(mcp_config_path=path)

def set_openai_api_key(api_key):
    """Set the OpenAI API key"""
    config_manager = ConfigManager()
    return config_manager.set(openai_api_key=api_key)

def get_openai_api_key():
    """Get the current OpenAI API key"""
    config_manager = ConfigManager()
    return config_manager.get("openai_api_key", "")

def enable_voice(enabled=True):
    """Enable or disable voice functionality"""
    config_manager = ConfigManager()
    return config_manager.set(voice=enabled)

def enable_devmode(enabled=True):
    """Enable or disable development mode"""
    config_manager = ConfigManager()
    return config_manager.set(devmode=enabled)

def set_db_clear(clear=True):
    """Set whether to clear database on startup"""
    config_manager = ConfigManager()
    return config_manager.set(db_clear=clear)

def enable_mcp(enabled=True):
    """Enable or disable MCP functionality"""
    config_manager = ConfigManager()
    return config_manager.set(use_mcp=enabled)

def get_config():
    """Get current configuration"""
    config_manager = ConfigManager()
    return config_manager.load_config()

def update_config(**kwargs):
    """Update multiple configuration values at once"""
    config_manager = ConfigManager()
    return config_manager.set(**kwargs)

# Example usage:
if __name__ == "__main__":
    # Test the configuration system
    print("Current config:", get_config())
    
    # Set OpenAI with API key
    set_openai(True, "gpt-4", "sk-your-api-key-here")
    print("After setting OpenAI with API key:", get_config())
    
    # Or set API key separately
    set_openai_api_key("sk-another-api-key")
    print("After updating API key:", get_config())
    
    # Switch to HackClub
    set_hackclub(True)
    print("After setting HackClub:", get_config())
    
    # Switch to Ollama
    set_ollama()
    print("After setting Ollama:", get_config())
    
    # Set RAG model to OpenAI
    set_rag_model("openai")
    print("After setting RAG model to OpenAI:", get_config())
    
    # Update multiple settings at once
    update_config(
        voice=True,
        devmode=True,
        rag_model="ollama",
        mcp_config_path="custom_mcp.json"
    )
    print("After bulk update:", get_config())