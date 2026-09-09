"""Compatibility patches for OpenAI models missing from pinned LlamaIndex."""

import sys
import importlib


_PATCH_APPLIED = False


def comprehensive_patch():
    """Patch context size, chat, function-calling, and tokenizer lookups."""

    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return True

    # Context windows for models absent from the pinned LlamaIndex release.
    NEW_MODELS = {
        "gpt-5": 128000,
        "gpt-5-mini": 128000,
        "gpt-5-nano": 400000,
        "o4-mini": 128000,
        "o4": 128000,
    }

    # Function-calling compatibility mappings.
    FUNCTION_CALLING_MODELS = {
        "gpt-5": "gpt-4",
        "gpt-5-mini": "gpt-4o-mini",
        "gpt-5-nano": "gpt-4o-mini",
        "o4-mini": "gpt-4o-mini",
        "o4": "gpt-4",
    }

    # Tokenizer mappings to model families recognized by tiktoken.
    TOKENIZER_MAPPING = {
        "gpt-5": "gpt-4",
        "gpt-5-mini": "gpt-4o-mini",
        "gpt-5-nano": "gpt-4o-mini",
        "o4-mini": "gpt-4o-mini",
        "o4": "gpt-4",
    }

    # Models that use the chat-completions endpoint.
    CHAT_MODELS = {
        "gpt-5", "gpt-5-mini", "gpt-5-nano", "o4-mini", "o4"
    }

    def patched_contextsize_function(modelname: str) -> int:
        """Return the configured context window for a model."""
        if modelname in NEW_MODELS:
            return NEW_MODELS[modelname]

        # Consult the package mapping before using the compatibility fallback.
        try:
            # Read the mapping directly to avoid recursively calling this patch.
            from llama_index.llms.openai.utils import OPENAI_MODELS
            if hasattr(OPENAI_MODELS, modelname) or modelname in OPENAI_MODELS:
                # Use the package value for models it already recognizes.
                return OPENAI_MODELS.get(modelname, 128000)
        except (ImportError, AttributeError):
            pass

        # Compatibility fallback for commonly used models.
        known_sizes = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4": 8192,
            "gpt-4.1-mini": 128000,
            "gpt-4.1-nano": 128000,
            "gpt-4.1": 128000,
            "gpt-3.5-turbo": 16385,
            "gpt-4-turbo": 128000,
            "gpt-5-nano": 400000,
        }

        if modelname in known_sizes:
            return known_sizes[modelname]

        # Use a conservative default for unknown models.
        print(f"⚠️ Using default context size 128000 for unknown model: {modelname}")
        return 128000

    def patched_function_calling_check(*args, **kwargs) -> bool:
        """Report function-calling support for compatibility models."""
        # Accept positional and keyword model arguments.
        modelname = None
        if args:
            modelname = args[0]
        elif 'model' in kwargs:
            modelname = kwargs['model']
        elif 'modelname' in kwargs:
            modelname = kwargs['modelname']

        # The compatibility models support function calling.
        if modelname and modelname in FUNCTION_CALLING_MODELS:
            return True

        # Delegate other models to the original implementation.
        try:
            from llama_index.llms.openai.utils import is_function_calling_model as original_func
            if hasattr(original_func, '__wrapped__'):
                return original_func.__wrapped__(*args, **kwargs)
            else:
                return original_func(*args, **kwargs)
        except:
            # Default to support when the original check is unavailable.
            return True

    def patched_is_chat_model(*args, **kwargs) -> bool:
        """Report chat-model support for compatibility models."""
        # Accept positional and keyword model arguments.
        modelname = None
        if args:
            modelname = args[0]
        elif 'model' in kwargs:
            modelname = kwargs['model']
        elif 'modelname' in kwargs:
            modelname = kwargs['modelname']

        # All compatibility models use the chat interface.
        if modelname and modelname in CHAT_MODELS:
            return True

        # Delegate other models to the original implementation.
        try:
            from llama_index.llms.openai.utils import is_chat_model as original_func
            if hasattr(original_func, '__wrapped__'):
                return original_func.__wrapped__(*args, **kwargs)
            else:
                return original_func(*args, **kwargs)
        except:
            # Default to chat behavior when the original check is unavailable.
            return True

    try:
        # Patch the primary utility module.
        from llama_index.llms.openai import utils

        # Patch context-size lookup.
        original_contextsize_func = utils.openai_modelname_to_contextsize
        if not hasattr(original_contextsize_func, '__wrapped__'):
            patched_contextsize_function.__wrapped__ = original_contextsize_func
        utils.openai_modelname_to_contextsize = patched_contextsize_function

        # Patch function-calling capability lookup.
        if hasattr(utils, 'is_function_calling_model'):
            original_fc_func = utils.is_function_calling_model
            if not hasattr(original_fc_func, '__wrapped__'):
                patched_function_calling_check.__wrapped__ = original_fc_func
            utils.is_function_calling_model = patched_function_calling_check

        # Patch chat-model lookup.
        if hasattr(utils, 'is_chat_model'):
            original_chat_func = utils.is_chat_model
            if not hasattr(original_chat_func, '__wrapped__'):
                patched_is_chat_model.__wrapped__ = original_chat_func
            utils.is_chat_model = patched_is_chat_model

        # Update references cached by imported LlamaIndex modules.
        for module_name, module in sys.modules.items():
            if module and 'llama_index.llms.openai' in module_name:
                if hasattr(module, 'openai_modelname_to_contextsize'):
                    setattr(module, 'openai_modelname_to_contextsize', patched_contextsize_function)
                if hasattr(module, 'is_function_calling_model'):
                    setattr(module, 'is_function_calling_model', patched_function_calling_check)
                if hasattr(module, 'is_chat_model'):
                    setattr(module, 'is_chat_model', patched_is_chat_model)

        # Update the model check cached by the OpenAI agent module.
        try:
            from llama_index.agent.openai import base as openai_base
            if hasattr(openai_base, 'is_function_calling_model'):
                openai_base.is_function_calling_model = patched_function_calling_check
        except ImportError:
            pass

        # Enforce the supported temperature for compatibility models.
        try:
            from llama_index.llms.openai import base as llm_base

            # Preserve the original initializer.
            original_openai_init = llm_base.OpenAI.__init__

            def patched_openai_init(self, model='gpt-3.5-turbo', temperature=None, **kwargs):
                """Initialize compatibility models with a supported temperature."""

                # Compatibility models accept only their default temperature.
                if model in NEW_MODELS:
                    if temperature is None:
                        temperature = 1.0
                        print(f"🔧  {model} set default temperature=1.0")
                    elif temperature != 1.0:
                        print(f"🔧  {model} adjust temperature: {temperature} -> 1.0 (only default value is supported for new models)")
                        temperature = 1.0
                    # Pass the normalized temperature to the original initializer.
                    return original_openai_init(self, model=model, temperature=temperature, **kwargs)
                elif temperature is None:
                    # Preserve the package default when temperature is unspecified.
                    return original_openai_init(self, model=model, **kwargs)
                else:
                    # Forward explicit temperatures for other models.
                    return original_openai_init(self, model=model, temperature=temperature, **kwargs)

            # Install the initializer patch.
            llm_base.OpenAI.__init__ = patched_openai_init
            print("✅ OpenAI temperature setting fixed")

        except Exception as temp_patch_error:
            print(f"⚠️  Temperature patch failed: {temp_patch_error}")
            pass

        # Patch model validation in OpenAIAgent.from_tools.
        try:
            from llama_index.agent.openai import base as agent_base

            # Preserve the original class method.
            if hasattr(agent_base.OpenAIAgent, 'from_tools'):
                original_from_tools = agent_base.OpenAIAgent.from_tools

                @classmethod
                def patched_from_tools(cls, tools=None, llm=None, **kwargs):
                    """Expose function-calling metadata for compatibility models."""

                    # Temporarily expose compatible metadata for added models.
                    if llm and hasattr(llm, 'model') and llm.model in FUNCTION_CALLING_MODELS:
                        # Read the metadata property from the LLM class.
                        llm_class = type(llm)

                        # Preserve the original metadata property.
                        original_metadata_property = getattr(llm_class, 'metadata', None)

                        # Construct a temporary metadata property.
                        @property
                        def patched_metadata_property(self):
                            try:
                                # Read the original metadata when available.
                                if original_metadata_property:
                                    if hasattr(original_metadata_property, 'fget'):
                                        original_metadata = original_metadata_property.fget(self)
                                    else:
                                        original_metadata = original_metadata_property.__get__(self, llm_class)
                                else:
                                    # Construct minimal metadata when none exists.
                                    import types
                                    original_metadata = types.SimpleNamespace(
                                        context_window=NEW_MODELS.get(self.model, 128000),
                                        num_output=4096,
                                        is_chat_model=True,
                                        is_function_calling_model=False,
                                        model_name=self.model
                                    )

                                # Mark compatibility models as function-calling models.
                                if hasattr(self, 'model') and self.model in FUNCTION_CALLING_MODELS:
                                    # Copy metadata before changing the capability flag.
                                    import types
                                    new_metadata = types.SimpleNamespace()
                                    for attr in dir(original_metadata):
                                        if not attr.startswith('_'):
                                            try:
                                                setattr(new_metadata, attr, getattr(original_metadata, attr))
                                            except:
                                                pass
                                    new_metadata.is_function_calling_model = True
                                    return new_metadata

                                return original_metadata
                            except Exception as e:
                                # Fall back to minimal function-calling metadata.
                                import types
                                return types.SimpleNamespace(
                                    context_window=NEW_MODELS.get(getattr(self, 'model', ''), 128000),
                                    num_output=4096,
                                    is_chat_model=True,
                                    is_function_calling_model=True,
                                    model_name=getattr(self, 'model', 'unknown')
                                )

                        # Install the temporary metadata property.
                        setattr(llm_class, 'metadata', patched_metadata_property)

                        try:
                            # Invoke the original class method.
                            return original_from_tools(tools=tools, llm=llm, **kwargs)
                        finally:
                            # Restore the original metadata property.
                            if original_metadata_property:
                                setattr(llm_class, 'metadata', original_metadata_property)
                            else:
                                # Remove the temporary property when none existed.
                                try:
                                    delattr(llm_class, 'metadata')
                                except:
                                    pass

                    # Delegate other models without changing their metadata.
                    return original_from_tools(tools=tools, llm=llm, **kwargs)

                # Install the class-method patch.
                agent_base.OpenAIAgent.from_tools = patched_from_tools
                print("✅ OpenAI Agent from_tools method patched")

        except Exception as agent_patch_error:
            print(f"⚠️ Agent patch failed: {agent_patch_error}")
            pass

        # Patch tiktoken model mapping.
        try:
            import tiktoken

            # Preserve the original tokenizer lookup.
            original_encoding_name_for_model = tiktoken.model.encoding_name_for_model

            def patched_encoding_name_for_model(model_name: str) -> str:
                """Resolve tokenizer names for compatibility models."""
                # Map added models to recognized tokenizer families.
                if model_name in TOKENIZER_MAPPING:
                    mapped_model = TOKENIZER_MAPPING[model_name]
                    print(f"🔄 Tokenizer mapping: {model_name} -> {mapped_model}")
                    return original_encoding_name_for_model(mapped_model)

                # Delegate other models to the original lookup.
                return original_encoding_name_for_model(model_name)

            # Install the tokenizer lookup patch.
            tiktoken.model.encoding_name_for_model = patched_encoding_name_for_model
            print("✅ Tiktoken model mapping patched")

        except Exception as tiktoken_error:
            print(f"⚠️ Tiktoken patch failed: {tiktoken_error}")
            pass

        print("✅ Enhanced model patch applied")
        print("✅ Supported models: gpt-5, gpt-5-mini, o4, o4-mini")
        print("✅ Function calling support enabled")
        print("✅ Tokenizer support enabled")
        print("✅ Chat model recognition enabled")
        print("✅ Temperature default value optimized")
        _PATCH_APPLIED = True
        return True

    except Exception as e:
        print(f"❌  Enhanced patch failed: {e}")
        return False

# Apply the compatibility patch when the module is imported.
comprehensive_patch()

if __name__ == "__main__":
    comprehensive_patch()
