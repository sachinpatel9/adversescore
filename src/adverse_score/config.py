import os
from dotenv import load_dotenv

def initialize_config() -> str:
    """
    Loads environment variables and validates the presence of all required API keys.
    Returns the openFDA API key for the AdverseScoreClient.
    """
    load_dotenv()
    
    fda_key = os.getenv('OPENFDA_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')

    # Fail Fast Validations
    if not fda_key:
        raise EnvironmentError("OPENFDA_API_KEY is not set in environment variables. Please set it in your .env file.")
    
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. The LangChain Agent requires this to execute.")

    return fda_key

# Execution block for testing the config directly
if __name__ == '__main__':
    key = initialize_config()
    import logging, json, sys
    logging.basicConfig(stream=sys.stderr, format='%(message)s')
    logging.info(json.dumps({"event": "config_initialized"}))