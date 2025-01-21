from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)
generator = pipeline("text-generation", model="gpt2-large")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["GET", "POST"])
def generate():
    if request.method == "GET":
        # Redirect to home to display the form
        return home()

    # Handling the POST request
    prompt = request.form["prompt"]
    if not prompt:
        return "Please enter a prompt"

    # Adjustable parameters for the text generation
    max_len = int(request.form.get("max_length", 150))
    temp = float(request.form.get("temperature", 0.7))

    gen_text = generator(
        prompt,
        max_length=max_len,
        num_return_sequences=1,
        temperature=temp,
    )

    final_result = gen_text[0]["generated_text"]
    return render_template(
        "index.html",
        prompt=prompt,
        result=final_result,
    )

if __name__ == "__main__":
    app.run(debug=True)
