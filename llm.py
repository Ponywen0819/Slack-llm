class LlmService():
    def __init__(self, factory):
        self.llm = factory.create_llm()

    def generate_text(self, prompt):
        return self.llm.generate(prompt)
    
# class LlmFactory():
#     def create_llm(self):
#         # Placeholder for actual LLM creation logic
#         return DummyLlm()