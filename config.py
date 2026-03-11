import os

from dotenv import load_dotenv

def initialize_config():
    '''
    Loads environment variables and validates presence of API key
    '''
    load_dotenv()
    api_key = os.getenv('OPENFDA_API_KEY')

    if not api_key:
        raise EnvironmentError('OPENFDA_API_KEY is not set in environment variables. Please set it in your .env file')
    print('Configuration initialized successfully. API key loaded.')
    return api_key

#execution 
if __name__ == '__main__':
    key = initialize_config()