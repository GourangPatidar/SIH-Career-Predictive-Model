from flask import Flask, request, jsonify
import openai

app = Flask(__name__)

# Replace 'YOUR_API_KEY' with your actual OpenAI API key
api_key = 'key Req'

# Initialize the OpenAI API client
openai.api_key = api_key

# Send a message to ChatGPT


def send_message(message, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ]
    )

    return response


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message')

        if not user_message:
            return jsonify({"error": "Please provide a 'message' field in the request."})

        response = send_message(user_message)
        return jsonify({"assistant_response": response['choices'][0]['message']['content']})
    except Exception as e:
        return jsonify({"error": str(e)})
    

if __name__ == '__main__':
    app.run(debug=True)
