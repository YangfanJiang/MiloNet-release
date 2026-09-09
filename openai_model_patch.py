"""OpenAI model compatibility for the pinned LlamaIndex release."""

def patch_openai_model_support():
    """Register models absent from the pinned LlamaIndex release."""
    try:
        # Import the module-level utilities that require compatibility updates.
        from llama_index.llms.openai import utils

        new_chat_models = {
            "gpt-5",
            "gpt-5.1-mini",
            "gpt-5-mini",
            "o4-mini",
            "o4",
        }
        utils.CHAT_MODELS.update({model: 128000 for model in new_chat_models})
        
        # Preserve the original context-size lookup for known models.
        original_openai_modelname_to_contextsize = utils.openai_modelname_to_contextsize
        
        def patched_openai_modelname_to_contextsize(modelname: str) -> int:
            """Return context sizes for both native and added models."""
            # Context sizes for models absent from the pinned LlamaIndex release.
            new_models = {
                "gpt-5": 128000,        # similar to gpt-4 context size
                "gpt-5.1-mini": 128000,  # similar to gpt-4o-mini context size
                "gpt-5-mini": 128000,   # similar to gpt-4o-mini context size
                "o4-mini": 128000,      # similar to gpt-4o-mini context size
                "o4": 128000,           # similar to gpt-4o context size
            }
            
            # Resolve added models before consulting the original mapping.
            if modelname in new_models:
                return new_models[modelname]
            
            # Delegate known models to the original function.
            try:
                return original_openai_modelname_to_contextsize(modelname)
            except ValueError:
                # Use the compatibility default when neither mapping has the model.
                print(f"⚠️ Unknown model '{modelname}', using default context size 128000")
                return 128000
        
        # Update the primary utility reference.
        utils.openai_modelname_to_contextsize = patched_openai_modelname_to_contextsize
        
        # Update any separately imported utility reference.
        try:
            import llama_index.llms.openai.utils as openai_utils
            openai_utils.openai_modelname_to_contextsize = patched_openai_modelname_to_contextsize
        except:
            pass
            
        # Update the reference cached by the base module when present.
        try:
            from llama_index.llms.openai import base
            if hasattr(base, 'openai_modelname_to_contextsize'):
                base.openai_modelname_to_contextsize = patched_openai_modelname_to_contextsize
        except:
            pass
            
        print("✅ OpenAI model support patch applied")
        print("✅ Now support: gpt-5, gpt-5.1-mini, gpt-5-mini, o4, o4-mini")
        
        return True
        
    except Exception as e:
        print(f"❌ patch failed: {e}")
        return False

if __name__ == "__main__":
    patch_openai_model_support()
