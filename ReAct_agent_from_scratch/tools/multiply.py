import json
from typing import Optional

def multiply(a: float, b: float) -> Optional[str]:
    """
    Multiply two numbers and return the result as a JSON string.

    Args:
        a (float): The first number.
        b (float): The second number.

    Returns:
        Optional[str]: A JSON string containing the inputs and the result.
    """
    try:
        result = a * b
        output = {
            "a": a,
            "b": b,
            "result": result
        }
        print(f"Multiplying {a} and {b}: {result}")
        return json.dumps(output, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"An error occurred during multiplication: {e}")
        return None

if __name__ == '__main__':
    # Example usage
    print(multiply(3, 4))