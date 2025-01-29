from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class DynamicRetrievalDecision:
    def __init__(self, model_name="distilbert-base-uncased", threshold=0.5):
        """
        Initialize the dynamic retrieval decision system.
        
        Models:
            - distilbert-base-uncased
            - microsoft/MiniLM-L12-H384-uncased
            - google/mobilebert-uncased

        Args:
            model_name (str): Name of the Hugging Face model to use.
            threshold (float): Probability threshold for determining sufficiency.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.threshold = threshold

    def retrieve(self, context, query):
        """
        Decide whether retrieval is needed based on context and query.

        Args:
            context (str): The context text available to the agent.
            query (str): The query text to evaluate.

        Returns:
            str: Decision indicating whether retrieval is needed.
        """
        # Combine context and query
        input_text = f"{context} [SEP] {query}"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Get model predictions
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        # Determine sufficiency
        sufficient_probability = probabilities[0][1].item()  # Assuming label 1 is "Sufficient"
        if sufficient_probability >= self.threshold:
            return True # f"Sufficient context. No retrieval needed. (Confidence: {sufficient_probability:.2f})"
        else:
            return False # f"Insufficient context. Retrieval needed. (Confidence: {sufficient_probability:.2f})"

"""
# Example usage
if __name__ == "__main__":
    # Initialize with a Hugging Face model
    decision_maker = DynamicRetrievalDecision(model_name="distilbert-base-uncased", threshold=0.6)

    # Example context and query
    context = "The Eiffel Tower is located in Paris."
    query = "Where is the Eiffel Tower located?"

    # Make a decision
    result = decision_maker.retrieve(context, query)
    print(result)
"""