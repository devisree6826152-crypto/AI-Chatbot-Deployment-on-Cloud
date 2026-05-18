from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

# Load FLAN-T5 model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    response = get_chat_response(msg)
    return response


def get_chat_response(text):
    
    text = text.lower()

    # Simple fallback replies
    if len(text.strip()) < 2:
        return "Please type something."

    if text in ["hi", "hello", "hey"]:
        return "Hello! How can I help you?"

    if "physics" in text:
        return "Physics is the study of matter, force, and energy."

    if "math" in text:
        return "Math is the study of numbers and calculations."

    # Default fallback
    return "I'm not sure what you mean by that."

    # Convert text to tokens
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


if __name__ == "__main__":
    app.run(debug=True)
