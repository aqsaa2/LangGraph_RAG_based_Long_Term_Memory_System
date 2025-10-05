# test_memory_extraction.py - Debug script to test memory extraction

import asyncio
import os
from langchain_core.messages import HumanMessage, AIMessage
from langmem import create_memory_store_manager
from memory_graph.configuration import Configuration, DEFAULT_MEMORY_CONFIGS

async def test_memory_extraction():
    """Test memory extraction with sample conversations."""
    
    # Set up configuration
    os.environ["USER_ID"] = "test_user_123"
    os.environ["MODEL"] = "gemini-2.0-flash"
    
    # Sample conversation messages
    test_messages = [
        HumanMessage(content="Hello, my name is Maria and I'm 28 years old."),
        AIMessage(content="Nice to meet you Maria! How can I help you today?"),
        HumanMessage(content="I'm a teacher and I love hiking on weekends. I also enjoy reading mystery novels."),
        AIMessage(content="That sounds wonderful! Teaching and hiking are both great activities.")
    ]
    
    print("=== Testing Memory Extraction ===")
    print(f"Messages to process: {len(test_messages)}")
    for i, msg in enumerate(test_messages):
        print(f"  {i}: {type(msg).__name__} - {msg.content[:50]}...")
    
    # Test each memory type
    for memory_config in DEFAULT_MEMORY_CONFIGS:
        print(f"\n--- Testing {memory_config.name} Memory Type ---")
        print(f"Update mode: {memory_config.update_mode}")
        print(f"System prompt: {memory_config.system_prompt[:100]}...")
        
        try:
            # Create store manager
            kwargs = {
                "enable_inserts": memory_config.update_mode in ["insert", "append"],
            }
            
            if memory_config.system_prompt:
                kwargs["instructions"] = memory_config.system_prompt
                
            print(f"Store manager kwargs: {kwargs}")
            
            store_manager = create_memory_store_manager(
                "gemini-2.0-flash",
                namespace=("memories", "test_user_123", memory_config.name),
                **kwargs,
            )
            
            # Test invocation
            manager_input = {
                "messages": test_messages,
                "max_steps": 1
            }
            
            config = {
                "configurable": {
                    "model": "gemini-2.0-flash",
                    "user_id": "test_user_123"
                }
            }
            
            print("Invoking store manager...")
            result = await store_manager.ainvoke(manager_input, config=config)
            
            print(f"Result type: {type(result)}")
            print(f"Result: {result}")
            
            if isinstance(result, AIMessage):
                print(f"AIMessage content: {result.content}")
                if hasattr(result, 'tool_calls') and result.tool_calls:
                    print(f"Tool calls found: {len(result.tool_calls)}")
                    for i, tc in enumerate(result.tool_calls):
                        print(f"  Tool call {i}: {tc}")
                else:
                    print("No tool calls found")
            elif isinstance(result, list):
                print(f"List result with {len(result)} items:")
                for i, item in enumerate(result):
                    print(f"  Item {i}: {type(item)} - {item}")
                    if isinstance(item, AIMessage) and hasattr(item, 'tool_calls'):
                        print(f"    Tool calls: {getattr(item, 'tool_calls', None)}")
            
        except Exception as e:
            print(f"ERROR testing {memory_config.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_memory_extraction())