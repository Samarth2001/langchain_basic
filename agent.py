import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()


def deepseek_query(question: str) -> str:
    """Direct query handler for DeepSeek-R1 through OpenRouter"""
    try:
        # Initialize model with required configuration
        model = ChatOpenAI(
            model="deepseek/deepseek-r1:free",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            max_tokens=512,
            temperature=0.7,
            default_headers={
                "HTTP-Referer": os.getenv("SITE_URL", "http://localhost:3000"),
                "X-Title": os.getenv("SITE_NAME", "Direct Query Handler"),
            },
        )

        # Simple direct prompt structure
        response = model.invoke(
            [
                HumanMessage(
                    content=f"""
                Answer the following question concisely and accurately.
                Question: {question}
                Answer in 1-2 sentences directly without formatting.
            """
                )
            ]
        )

        return response.content.strip()

    except Exception as e:
        return f"Error processing request: {str(e)}"


# Example usage
if __name__ == "__main__":
    question = "What is the capital of France?"
    result = deepseek_query(question)
    print(f"Question: {question}")
    print(f"Answer: {result}")
