"""Quick smoke tests for the conversation generator."""

import asyncio
import os

def check_api_keys():
    keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Google": os.getenv("GOOGLE_API_KEY")
    }
    
    print("API key status:")
    for name, key in keys.items():
        status = "set" if key else "not set"
        print(f"  {name}: {status}")
    
    return keys


async def test_openai():
    """Test OpenAI API connection."""
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Say 'API test successful' and nothing else."}],
        max_tokens=20
    )
    return response.choices[0].message.content


async def test_anthropic():
    """Test Anthropic API connection."""
    import anthropic
    
    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=20,
        messages=[{"role": "user", "content": "Say 'API test successful' and nothing else."}]
    )
    return response.content[0].text


def test_dataset_loading():
    """Test HuggingFace dataset loading."""
    from datasets import load_dataset
    
    print("\nLoading HealthCareMagic dataset (first 3 rows)...")
    dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
    
    print(f"Dataset loaded: {len(dataset)} total rows")
    print(f"Columns: {dataset.column_names}")
    
    print("\nSample scenario")
    sample = dataset[0]
    print(f"\nInstruction: {sample['instruction'][:100]}...")
    print(f"\nPatient Input: {sample['input'][:300]}...")
    print(f"\nDoctor Output: {sample['output'][:300]}...")
    
    return dataset


async def test_mini_conversation():
    """Generate a mini 4-turn conversation to verify the pipeline."""
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI()
    
    physician_system = """You are a physician in a telemedicine consultation. 
    Ask relevant questions and provide medical guidance. Be concise."""
    
    patient_system = """You are a patient describing symptoms. 
    Original complaint: Headache for 3 days, worse in the morning.
    Respond naturally and briefly (1-2 sentences)."""
    
    turns = []
    
    patient_initial = "I've had this headache for about 3 days now. It's worse when I wake up."
    turns.append({"role": "patient", "content": patient_initial})
    print(f"\nPatient (Turn 0): {patient_initial}")

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": physician_system},
            {"role": "user", "content": f"Patient says: {patient_initial}\n\nRespond as the physician."}
        ],
        max_tokens=200
    )
    physician_1 = response.choices[0].message.content
    turns.append({"role": "physician", "content": physician_1})
    print(f"\nPhysician (Turn 1): {physician_1}")

    history = f"Patient: {patient_initial}\nPhysician: {physician_1}"
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": patient_system},
            {"role": "user", "content": f"Conversation so far:\n{history}\n\nRespond as the patient."}
        ],
        max_tokens=100
    )
    patient_2 = response.choices[0].message.content
    turns.append({"role": "patient", "content": patient_2})
    print(f"\nPatient (Turn 2): {patient_2}")

    history += f"\nPatient: {patient_2}"
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": physician_system},
            {"role": "user", "content": f"Conversation so far:\n{history}\n\nRespond as the physician."}
        ],
        max_tokens=200
    )
    physician_3 = response.choices[0].message.content
    turns.append({"role": "physician", "content": physician_3})
    print(f"\nPhysician (Turn 3): {physician_3}")
    
    return turns


async def main():
    print("Medical conversation generation test")
    keys = check_api_keys()

    try:
        test_dataset_loading()
        print("\nDataset loading: OK")
    except Exception as e:
        print(f"\nDataset loading: FAIL ({e})")
        return

    if keys["OpenAI"]:
        print("\nTesting OpenAI API...")
        try:
            result = await test_openai()
            print(f"OpenAI API: OK ({result})")
        except Exception as e:
            print(f"OpenAI API: FAIL ({e})")
    
    if keys["Anthropic"]:
        print("\nTesting Anthropic API...")
        try:
            result = await test_anthropic()
            print(f"Anthropic API: OK ({result})")
        except Exception as e:
            print(f"Anthropic API: FAIL ({e})")

    if keys["OpenAI"]:
        print("\nGenerating test conversation (4 turns)")
        try:
            turns = await test_mini_conversation()
            print("\nConversation generation: OK")
            print(f"  Generated {len(turns)} turns")
        except Exception as e:
            print(f"\nConversation generation: FAIL ({e})")

    print("\nTest complete")
    print("Run full generation with:")
    print("  python generate_conversations.py --num_scenarios 100 --turns 8")


if __name__ == "__main__":
    asyncio.run(main())
