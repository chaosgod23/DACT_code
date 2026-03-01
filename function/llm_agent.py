from openai import OpenAI


class LLMAgent:
    def __init__(self, args):
        self.args = args
        self.client = OpenAI(
            base_url=args['base_url'],
            api_key=args['api_key'],
        )

    def get_embedding(self, text):
        return self.client.embeddings.create(
            model=self.args['model'],
            input=text,
        )
